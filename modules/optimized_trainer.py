import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import gc

from modules import ANN

class Trainer:
    def __init__(self, n_inputs, image_size, n_epochs, device, model_save_dir, lr=1e-4, batch_size=16, 
                 patience=20, tolerance=0.15, min_epochs=150):
        """
        Initialize the Trainer with adaptive early stopping.
        
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
        patience : int
            Number of epochs to wait for improvement before early stopping
        tolerance : float
            Relative tolerance for loss variation
        min_epochs : int
            Minimum number of epochs before checking convergence
        """
        self.n_inputs = n_inputs
        self.image_size = image_size
        self.n_epochs = n_epochs
        self.device = device
        self.model_save_dir = Path(model_save_dir)
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.tolerance = tolerance
        self.min_epochs = min_epochs

    def check_convergence(self, loss_history, patience, tolerance):
        """
        Check if the loss has converged using relative metrics.
        
        Parameters:
        -----------
        loss_history : list
            List of loss values
        patience : int
            Number of epochs to check for convergence
        tolerance : float
            Relative tolerance for loss variation
            
        Returns:
        --------
        bool : True if converged, False otherwise
        """
        if len(loss_history) < patience:
            return False
            
        recent_losses = loss_history[-patience:]
        mean_loss = np.mean(recent_losses)
        
        # Calculate relative deviations
        relative_deviations = [abs(loss - mean_loss) / mean_loss for loss in recent_losses]
        max_relative_deviation = max(relative_deviations)
        
        # Check if relative deviations are within tolerance
        converged = max_relative_deviation <= tolerance
        
        if converged:
            print(f"\nEarly stopping triggered:")
            print(f"Mean loss: {mean_loss:.4f}")
            print(f"Maximum relative deviation: {max_relative_deviation:.4%}")
            print(f"Recent losses: {', '.join(f'{loss:.4f}' for loss in recent_losses)}")
            
        return converged

    def train(self, spk_in, train_images):
        """
        Train the network with adaptive early stopping.
        
        Parameters:
        -----------
        spk_in : torch.Tensor [n_batch, n_neurons, n_trials, time_steps]
            Input spike trains
        train_images : torch.Tensor [n_batch, height, width]
            Target images
            
        Returns:
        --------
        torch.Tensor : Loss history
        torch.Tensor : Final output
        ANN : Trained network
        """
        try:
            network = ANN(n_inputs=self.n_inputs, image_size=self.image_size)
            optimizer = optim.AdamW(network.parameters(), lr=self.lr, weight_decay=0.01)
            
            dataset = torch.utils.data.TensorDataset(spk_in, train_images)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            loss_hist = []
            best_loss = float('inf')
            
            print(f"Starting training with max {self.n_epochs} epochs...")
            print(f"Early stopping setup:")
            print(f"- Minimum epochs: {self.min_epochs}")
            print(f"- Patience: {self.patience} epochs")
            print(f"- Relative tolerance: {self.tolerance:.2%}")
            
            for epoch in range(self.n_epochs):
                network.train()
                epoch_loss = 0
                batch_count = 0
                
                for batch_spikes, batch_images in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    output = network(batch_spikes)
                    loss = network.loss_function(output, batch_images)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        gc.collect()
                
                avg_loss = epoch_loss / batch_count
                loss_hist.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    state_dict = {k: v.clone() for k, v in network.state_dict().items()}
                    torch.save(state_dict, self.model_save_dir / 'best_model.pt')
                
                if (epoch + 1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_loss:.4f}')
                    
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': {k: v.clone() for k, v in network.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }
                    torch.save(checkpoint, self.model_save_dir / f'checkpoint_epoch_{epoch+1}.pt')
                
                # Only check convergence after minimum epochs
                if epoch >= self.min_epochs:
                    if self.check_convergence(loss_hist, self.patience, self.tolerance):
                        print(f"\nTraining stopped early at epoch {epoch + 1}")
                        print(f"Loss has stabilized around {np.mean(loss_hist[-self.patience:]):.4f}")
                        break
            
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
        
        Parameters:
        -----------
        network : ANN
            Trained network to evaluate
        val_spk_in : torch.Tensor [n_batch, n_neurons, n_trials, time_steps]
            Validation spike trains
        val_images : torch.Tensor [n_batch, height, width]
            Validation target images
            
        Returns:
        --------
        float : Average validation loss
        torch.Tensor : Network outputs for validation data
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

