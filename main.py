import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import gc

os.environ['NNPACK_DISABLE'] = '1'  # Disable NNPACK
torch.backends.nnpack.enabled = False

from modules import ANN, Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Enable garbage collection
        gc.enable()
        
        # Set up directories
        logger.info("Setting up directories...")
        data_dir = Path('data')
        model_save_dir = Path('model')
        output_dir = Path('outputs')
        
        # Create directories if they don't exist
        for dir_path in [model_save_dir, output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load data with error handling
        logger.info("Loading data...")
        try:
            train_spikes = torch.load(data_dir / 'train_spikes_gratings.pt', weights_only=True).float()
            val_spikes = torch.load(data_dir / 'val_spikes_gratings.pt', weights_only=True).float()
            train_images = torch.load(data_dir / 'train_images_gratings.pt', weights_only=True).float()
            val_images = torch.load(data_dir / 'val_images_gratings.pt', weights_only=True).float()
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            return
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return
        
        # Print data shapes for verification
        logger.info("Data shapes:")
        logger.info(f"Train spikes: {train_spikes.shape}")
        logger.info(f"Train images: {train_images.shape}")
        logger.info(f"Val spikes: {val_spikes.shape}")
        logger.info(f"Val images: {val_images.shape}")
        
        # Data normalization
        logger.info("Normalizing data...")
        train_images = train_images / train_images.max()
        val_images = val_images / val_images.max()
        
        # Verify data range
        logger.info(f"Train images range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        logger.info(f"Val images range: [{val_images.min():.3f}, {val_images.max():.3f}]")
        
        # Set device
        device = torch.device("cpu")  # Force CPU usage
        logger.info(f'Using device: {device}')
        
        # Set training parameters
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        batch_size = 16
        n_epochs = 50
        
        # Set hyperparameters
        n_inputs = train_spikes.shape[0]  # number of neurons
        image_size = train_images.shape[1]  # assuming square images
        lr = 5e-4
        
        # Prepare input spikes
        spk_in = train_spikes.permute(1, 0, 2, 3)  # [batch, neurons, trials, time]
        val_spk_in = val_spikes.permute(1, 0, 2, 3)
        
        logger.info("Training shapes:")
        logger.info(f"Input spikes: {spk_in.shape}")
        logger.info(f"Target images: {train_images.shape}")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            n_inputs=n_inputs,
            image_size=image_size,
            n_epochs=n_epochs,
            device=device,
            model_save_dir=model_save_dir,
            lr=lr,
            batch_size=batch_size
        )
        
        # Training phase
        logger.info("Starting training...")
        loss_hist, decoded_image, network = trainer.train(spk_in, train_images)
        
        if loss_hist is not None:
            logger.info("Training completed successfully!")
            
            # Evaluation phase
            logger.info("Evaluating on validation set...")
            val_loss, decoded_image_val = trainer.eval(network, val_spk_in, val_images)
            
            # Save results
            logger.info("Saving results...")
            try:
                torch.save(loss_hist, output_dir / 'loss_hist_train.pt')
                torch.save(val_loss, output_dir / 'loss_val.pt')
                torch.save(decoded_image_val, output_dir / 'decoded_image_val.pt')
                
                # Plot and save training loss
                plt.figure(figsize=(10, 5))
                plt.plot(loss_hist.cpu().numpy())
                plt.title('Training Loss Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.savefig(output_dir / 'training_loss.png')
                plt.close()
                
                logger.info(f"Final validation loss: {val_loss:.6f}")
                logger.info(f"Results saved in: {output_dir}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        else:
            logger.error("Training failed!")
            
    except RuntimeError as e:
        logger.error(f"Runtime error occurred: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    finally:
        # Final cleanup
        gc.collect()
        logger.info("Process completed.")

if __name__ == "__main__":
    main()