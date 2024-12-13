import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import gc

from modules import ANN

class Trainer:
    def __init__(self, n_inputs, image_size, n_epochs, device, model_save_dir, lr=1e-4, batch_size=16):
        """
        Initialize the Trainer for the temporal ANN network.
        
        Parameters:
        -----------
        n_inputs : int
            Number of input neurons
        image_size : int
            Size of the output images
        n_epochs : int
            Number of training epochs
        device : torch.device
            Device to run training on (CPU)
        model_save_dir : Path
            Directory to save model checkpoints
        lr : float
            Learning rate
        batch_size : int
            Batch size for training
        """
        self.n_inputs = n_inputs
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.device = device
        self.model_save_dir = Path(model_save_dir)
        self.lr = lr
        self.batch_size = batch_size

    def train(self, spk_in, train_images):
        """
        Train the network.
        
        Parameters:
        -----------
        spk_in : torch.Tensor [n_batch, n_neurons, n_trials, time_steps]
            Input spike trains
        train_images : torch.Tensor [n_batch, height, width]
            Target images
        """
        try:
            # Initialize network
            network = ANN(n_inputs=self.n_inputs, image_size=self.image_size)
            
            # Use AdamW optimizer
            optimizer = optim.AdamW(network.parameters(), lr=self.lr, weight_decay=0.01)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(spk_in, train_images)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # CPU optimization
            )
            
            loss_hist = []
            best_loss = float('inf')
            
            print(f"Starting training with {self.n_epochs} epochs...")
            
            for epoch in range(self.n_epochs):
                network.train()
                epoch_loss = 0
                batch_count = 0
                
                for batch_spikes, batch_images in train_loader:
                    # Clear gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    output = network(batch_spikes)
                    loss = network.loss_function(output, batch_images)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Clean up memory
                    if batch_count % 10 == 0:
                        gc.collect()
                
                # Calculate average loss
                avg_loss = epoch_loss / batch_count
                loss_hist.append(avg_loss)
                
                # Save best model with proper state dict handling
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Clone state dict to avoid memory issues
                    state_dict = {k: v.clone() for k, v in network.state_dict().items()}
                    torch.save(state_dict, self.model_save_dir / 'best_model.pt')
                
                # Print progress
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_loss:.4f}')
                    
                # Regular checkpointing
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': {k: v.clone() for k, v in network.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }
                    torch.save(checkpoint, self.model_save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
            # Load best model for final output
            checkpoint = torch.load(self.model_save_dir / 'best_model.pt')
            network.load_state_dict({k: v.clone() for k, v in checkpoint.items()})
            
            return torch.tensor(loss_hist), output, network
            
        except Exception as e:
            print(f"Training error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def eval(self, network, val_spk_in, val_images):
        """
        Evaluate the network on validation data.
        """
        try:
            network.eval()
            
            val_dataset = torch.utils.data.TensorDataset(val_spk_in, val_images)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            total_loss = 0
            outputs = []
            
            with torch.no_grad():
                for batch_spikes, batch_images in val_loader:
                    output = network(batch_spikes)
                    loss = network.loss_function(output, batch_images)
                    
                    total_loss += loss.item()
                    outputs.append(output)
            
            avg_loss = total_loss / len(val_loader)
            all_outputs = torch.cat(outputs, dim=0)
            
            return avg_loss, all_outputs
            
        except Exception as e:
            print(f"Evaluation error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None