import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from scipy.ndimage import center_of_mass

from modules import ANN
os.environ['NNPACK_DISABLE'] = '1'  # Disable NNPACK

class ModelEvaluator:
    def __init__(self, model, val_spikes, val_images, save_dir='metrics'):
        self.model = model
        self.val_spikes = val_spikes
        self.val_images = val_images
        self.save_dir = Path(save_dir)
        
        # Create directory structure
        self.metrics_dir = self.save_dir / 'metrics'
        self.plots_dir = self.save_dir / 'plots'
        self.samples_dir = self.save_dir / 'sample_predictions'
        
        for directory in [self.metrics_dir, self.plots_dir, self.samples_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def compute_position_error(self, pred, target):
        """Compute distance between predicted and actual bar centers"""
        pred_binary = (pred > 0.5).float()
        target_binary = (target != 116.0).float()
        
        # Convert to numpy for center_of_mass calculation
        pred_np = pred_binary.cpu().numpy()
        target_np = target_binary.cpu().numpy()
        
        pred_center = center_of_mass(pred_np)
        target_center = center_of_mass(target_np)
        
        return np.sqrt(
            (pred_center[0] - target_center[0])**2 + 
            (pred_center[1] - target_center[1])**2
        )
    
    def compute_classification_metrics(self, pred, target, threshold=0.5):
        """Compute binary classification metrics"""
        pred_binary = (pred > threshold).float()
        target_binary = (target != 116.0).float()
    
        pred_flat = pred_binary.flatten().cpu().numpy()
        target_flat = target_binary.flatten().cpu().numpy()
        pred_prob_flat = pred.flatten().cpu().numpy()
    
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average='binary'
        )
        accuracy = accuracy_score(target_flat, pred_flat)
    
        # Check for single-class targets
        if len(np.unique(target_flat)) > 1:
            auc = roc_auc_score(target_flat, pred_prob_flat)
        else:
            auc = None  # AUC not defined for single-class targets
    
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def evaluate(self):
        """Run complete evaluation"""
        self.model.eval()
        metrics = {
            'position_errors': [],
            'center_activations': [],
            'mse_values': [],
            'classification_metrics': [],
            'center_classification_metrics': []
        }
        
        with torch.no_grad():
            # Process all validation samples
            outputs = self.model(self.val_spikes)
            
            # Compute metrics for each sample
            for i in range(len(outputs)):
                output = outputs[i]
                target = self.val_images[i]
                
                # Position error
                pos_error = self.compute_position_error(output, target)
                metrics['position_errors'].append(pos_error)
                
                # Center activation
                center_activation = output[140:180, 140:180].mean().item()
                metrics['center_activations'].append(center_activation)
                
                # MSE
                mse = F.mse_loss(output, target).item()
                metrics['mse_values'].append(mse)
                
                # Classification metrics
                full_metrics = self.compute_classification_metrics(output, target)
                center_metrics = self.compute_classification_metrics(
                    output[140:180, 140:180],
                    target[140:180, 140:180]
                )
                metrics['classification_metrics'].append(full_metrics)
                metrics['center_classification_metrics'].append(center_metrics)
                
                # Save sample predictions (first 10 samples)
                if i < 10:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    ax1.imshow(output.cpu(), cmap='gray')
                    ax1.set_title('Predicted')
                    ax2.imshow(target.cpu(), cmap='gray')
                    ax2.set_title('Ground Truth')
                    plt.savefig(self.samples_dir / f'sample_{i}.png')
                    plt.close()
            
            # Calculate MSE-based metrics
            mse_values = np.array(metrics['mse_values'])
            mse_threshold = np.mean(mse_values) + np.std(mse_values)
            mse_accuracy = np.mean(mse_values < mse_threshold)
            
            # Generate distribution plots
            
            # 1. Position Errors
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics['position_errors'])
            plt.title('Distribution of Position Errors')
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Count')
            plt.savefig(self.plots_dir / 'position_errors_dist.png')
            plt.close()
            
            # 2. Center Activations
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics['center_activations'])
            plt.title('Distribution of Center Region Activations')
            plt.xlabel('Average Activation')
            plt.ylabel('Count')
            plt.savefig(self.plots_dir / 'center_activations_dist.png')
            plt.close()
            
            # 3. MSE Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(metrics['mse_values'])
            plt.title('Distribution of MSE Values')
            plt.xlabel('MSE')
            plt.ylabel('Count')
            plt.savefig(self.plots_dir / 'mse_dist.png')
            plt.close()
            
            # Compute summary statistics
            summary = {
                'position_error': {
                    'mean': np.mean(metrics['position_errors']),
                    'std': np.std(metrics['position_errors'])
                },
                'center_activation': {
                    'mean': np.mean(metrics['center_activations']),
                    'std': np.std(metrics['center_activations'])
                },
                'mse': {
                    'mean': np.mean(metrics['mse_values']),
                    'std': np.std(metrics['mse_values']),
                    'min': np.min(metrics['mse_values']),
                    'max': np.max(metrics['mse_values']),
                    'range': np.max(metrics['mse_values']) - np.min(metrics['mse_values']),
                    'accuracy': mse_accuracy
                },
                'classification': {
                    'accuracy': np.mean([m['accuracy'] for m in metrics['classification_metrics']]),
                    'f1': np.mean([m['f1'] for m in metrics['classification_metrics']]),
                    'auc': np.mean([m['auc'] for m in metrics['classification_metrics'] if m['auc'] is not None])
                },
                'center_classification': {
                    'accuracy': np.mean([m['accuracy'] for m in metrics['center_classification_metrics']]),
                    'f1': np.mean([m['f1'] for m in metrics['center_classification_metrics']]),
                    'auc': np.mean([m['auc'] for m in metrics['center_classification_metrics'] if m['auc'] is not None])
                }
            }
            
            # Save summary metrics
            with open(self.metrics_dir / 'summary.txt', 'w') as f:
                for category, values in summary.items():
                    f.write(f"\n{category.upper()}:\n")
                    for metric, value in values.items():
                        if isinstance(value, float):
                            f.write(f"{metric}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
                
                # Add MSE distribution information
                f.write("\nMSE DISTRIBUTION PERCENTILES:\n")
                percentiles = [25, 50, 75, 90, 95, 99]
                for p in percentiles:
                    value = np.percentile(mse_values, p)
                    f.write(f"{p}th percentile: {value:.4f}\n")
            
            return summary, metrics

def load_model_and_data():
    # Set up paths
    model_path = Path('model') / 'best_model.pt'
    data_path = Path('data')
    
    # Load validation data
    val_spikes = torch.load(data_path / 'val_spikes_gratings.pt').float()
    val_images = torch.load(data_path / 'val_images_gratings.pt').float()
    
    val_spikes = val_spikes.permute(1, 0, 2, 3)

    # Initialize model with same parameters as training
    model = ANN(n_inputs=26, image_size=320)
    
    # Load model state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, val_spikes, val_images

if __name__ == "__main__":
    # Load everything and run evaluation
    model, val_spikes, val_images = load_model_and_data()
    evaluator = ModelEvaluator(model, val_spikes, val_images)
    summary, metrics = evaluator.evaluate()

    # Print summary
    print("\nEvaluation Summary:")
    for category, values in summary.items():
        print(f"\n{category.upper()}:")
        for metric, value in values.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
