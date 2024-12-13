import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ANN(nn.Module):
    def __init__(self, n_inputs=26, image_size=320):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.image_size = image_size
        
        # Temporal feature extraction
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(n_inputs, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 15, 128),
            nn.ReLU(inplace=True)
        )
        
        # Separate predictors
        self.position_predictor = nn.Linear(128, 2)  # x, y
        self.orientation_predictor = nn.Linear(128, 1)  # angle
        
        # Register enhanced sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Constants
        self.bar_width = 15.0
        self.bar_length = 40.0
        
        self.grid_initialized = False
        self.current_angles = None
        self._init_weights()
    
    def _init_weights(self):
        # Standard initialization for feature extractors
        for m in self.temporal_encoder.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.feature_processor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        
        # Initialize position predictor with center bias
        nn.init.normal_(self.position_predictor.weight, std=0.01)
        nn.init.constant_(self.position_predictor.bias, 0.5)
        
        # Initialize orientation predictor with horizontal bias
        nn.init.normal_(self.orientation_predictor.weight, std=0.005)  # Smaller for tighter distribution
        nn.init.zeros_(self.orientation_predictor.bias)  # Bias toward horizontal
    
    def _initialize_grid(self, device):
        if not self.grid_initialized:
            y, x = torch.meshgrid(
                torch.arange(self.image_size, dtype=torch.float32, device=device),
                torch.arange(self.image_size, dtype=torch.float32, device=device),
                indexing='ij'
            )
            self.grid_y = y
            self.grid_x = x
            self.grid_initialized = True

    def get_target_orientation(self, target):
        """Enhanced orientation extraction focusing on gradient direction"""
        # Get gradients
        grad_x = F.conv2d(target.unsqueeze(1), self.sobel_x, padding=1)
        grad_y = F.conv2d(target.unsqueeze(1), self.sobel_y, padding=1)
        
        # Focus on center region where gradient is strongest
        center = slice(140, 180)
        grad_x_center = grad_x[:, :, center, center]
        grad_y_center = grad_y[:, :, center, center]
        
        # Compute gradient magnitude for weighting
        grad_magnitude = torch.sqrt(grad_x_center**2 + grad_y_center**2)
        weights = grad_magnitude / (grad_magnitude.sum() + 1e-6)
        
        # Weighted average of gradients
        avg_grad_x = (grad_x_center * weights).sum(dim=(1, 2, 3))
        avg_grad_y = (grad_y_center * weights).sum(dim=(1, 2, 3))
        
        # Get angle and normalize to [-0.5, 0.5] for horizontal bias
        angle = torch.atan2(avg_grad_y, avg_grad_x)
        return angle / (2 * torch.pi)  # Normalize to [-0.5, 0.5]

    def preprocess_spikes(self, x):
        x = x.float()
        x = x.mean(dim=2)  # Average across trials
        x = x[..., 90:105]  # Critical window
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
        return x

    def generate_binary_bar(self, centers, angles):
        self._initialize_grid(centers.device)
        batch_size = centers.shape[0]
        output = []
        
        for i in range(batch_size):
            # Strong center bias
            center_x = self.image_size/2 + (centers[i, 0] - self.image_size/2) * 0.1
            center_y = self.image_size/2 + (centers[i, 1] - self.image_size/2) * 0.1
            
            dx = self.grid_x - center_x
            dy = self.grid_y - center_y
            
            # Convert normalized angle to radians with restricted range
            angle_rad = angles[i] * torch.pi  # Only allow [-pi/2, pi/2]
            cos_theta = torch.cos(angle_rad)
            sin_theta = torch.sin(angle_rad)
            
            x_rot = dx * cos_theta + dy * sin_theta
            y_rot = -dx * sin_theta + dy * cos_theta
            
            # Sharper bar edges
            x_mask = torch.sigmoid(-(x_rot.abs() - self.bar_length/2) * 35)
            y_mask = torch.sigmoid(-(y_rot.abs() - self.bar_width/2) * 35)
            bar_mask = torch.clamp(x_mask * y_mask, 0, 1)
            
            output.append(bar_mask)
        
        return torch.stack(output)

    def forward(self, x):
        # Extract features
        x = self.preprocess_spikes(x)
        features = self.temporal_encoder(x)
        shared_features = self.feature_processor(features)
        
        # Predict parameters with horizontal bias
        centers = torch.sigmoid(self.position_predictor(shared_features)) * self.image_size
        raw_angles = self.orientation_predictor(shared_features)
        angles = torch.tanh(raw_angles) * 0.5  # Restrict angle range
        
        self.current_angles = angles
        
        return self.generate_binary_bar(centers, angles)

    def loss_function(self, pred, target):
        # Extract target orientation
        target_angle = self.get_target_orientation(target)
        pred_angle = self.current_angles.squeeze(-1)
        
        # Basic shape loss
        pred = torch.clamp(pred, 0, 1)
        binary_target = ((target != 116.0).float()).clamp(0, 1)
        basic_loss = F.binary_cross_entropy(pred, binary_target)
        
        # Enhanced orientation matching
        angle_loss = (
            3.0 * F.mse_loss(torch.cos(pred_angle * 2 * torch.pi), 
                            torch.cos(target_angle * 2 * torch.pi)) +
            F.mse_loss(torch.sin(pred_angle * 2 * torch.pi), 
                       torch.sin(target_angle * 2 * torch.pi))
        )
        
        # Center region emphasis
        center_region = pred[:, 140:180, 140:180]
        target_region = binary_target[:, 140:180, 140:180]
        center_loss = F.binary_cross_entropy(center_region, target_region)
        
        # Horizontal bias regularization
        horizontal_bias = torch.abs(pred_angle).mean()  # Encourage smaller angles
        
        return basic_loss + 8.0 * angle_loss + 2.0 * center_loss + horizontal_bias

# class ANN(nn.Module):
#     def __init__(self, n_inputs=26, image_size=320):
#         super().__init__()
        
#         self.n_inputs = n_inputs
#         self.image_size = image_size
        
#         # Temporal feature extraction
#         self.temporal_encoder = nn.Sequential(
#             nn.Conv1d(n_inputs, 32, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
        
#         # Feature processing
#         self.feature_processor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 15, 128),
#             nn.ReLU(inplace=True)
#         )
        
#         # Separate predictors
#         self.position_predictor = nn.Linear(128, 2)  # x, y
#         self.orientation_predictor = nn.Linear(128, 1)  # angle
        
#         # Register sobel kernels as buffers
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         self.register_buffer('sobel_x', sobel_x)
#         self.register_buffer('sobel_y', sobel_y)
        
#         # Constants
#         self.bar_width = 15.0
#         self.bar_length = 40.0
        
#         self.grid_initialized = False
#         self._init_weights()
        
#         # Store intermediate values
#         self.current_angles = None
    
#     def _init_weights(self):
#         # Initialize feature extractors
#         for m in self.temporal_encoder.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         for m in self.feature_processor.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.zeros_(m.bias)
        
#         # Initialize position predictor to center bias
#         nn.init.normal_(self.position_predictor.weight, std=0.01)
#         nn.init.constant_(self.position_predictor.bias, 0.5)
        
#         # Initialize orientation predictor conservatively
#         nn.init.normal_(self.orientation_predictor.weight, std=0.01)
#         nn.init.zeros_(self.orientation_predictor.bias)
    
#     def _initialize_grid(self, device):
#         if not self.grid_initialized:
#             y, x = torch.meshgrid(
#                 torch.arange(self.image_size, dtype=torch.float32, device=device),
#                 torch.arange(self.image_size, dtype=torch.float32, device=device),
#                 indexing='ij'
#             )
#             self.grid_y = y
#             self.grid_x = x
#             self.grid_initialized = True

#     def get_target_orientation(self, target):
#         """Extract orientation from target image gradient"""
#         # Get gradients
#         grad_x = F.conv2d(target.unsqueeze(1), self.sobel_x, padding=1)
#         grad_y = F.conv2d(target.unsqueeze(1), self.sobel_y, padding=1)
        
#         # Focus on center region
#         center = slice(140, 180)
#         grad_x_center = grad_x[:, :, center, center]
#         grad_y_center = grad_y[:, :, center, center]
        
#         # Compute dominant orientation
#         avg_grad_x = grad_x_center.mean(dim=(1, 2, 3))
#         avg_grad_y = grad_y_center.mean(dim=(1, 2, 3))
        
#         # Get angle and normalize
#         angle = torch.atan2(avg_grad_y, avg_grad_x)
#         return angle / torch.pi  # Normalize to [-1, 1]

#     def preprocess_spikes(self, x):
#         x = x.float()
#         x = x.mean(dim=2)  # Average across trials
#         x = x[..., 90:105]  # Critical window
#         x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
#         return x

#     def generate_binary_bar(self, centers, angles):
#         self._initialize_grid(centers.device)
#         batch_size = centers.shape[0]
#         output = []
        
#         for i in range(batch_size):
#             # Strong center bias
#             center_x = self.image_size/2 + (centers[i, 0] - self.image_size/2) * 0.1
#             center_y = self.image_size/2 + (centers[i, 1] - self.image_size/2) * 0.1
            
#             dx = self.grid_x - center_x
#             dy = self.grid_y - center_y
            
#             # Convert normalized angle back to radians
#             angle_rad = angles[i] * torch.pi
#             cos_theta = torch.cos(angle_rad)
#             sin_theta = torch.sin(angle_rad)
            
#             x_rot = dx * cos_theta + dy * sin_theta
#             y_rot = -dx * sin_theta + dy * cos_theta
            
#             # Sharp bar mask
#             x_mask = torch.sigmoid(-(x_rot.abs() - self.bar_length/2) * 30)
#             y_mask = torch.sigmoid(-(y_rot.abs() - self.bar_width/2) * 30)
#             bar_mask = torch.clamp(x_mask * y_mask, 0, 1)
            
#             output.append(bar_mask)
        
#         return torch.stack(output)

#     def forward(self, x):
#         # Extract features
#         x = self.preprocess_spikes(x)
#         features = self.temporal_encoder(x)
#         shared_features = self.feature_processor(features)
        
#         # Predict parameters
#         centers = torch.sigmoid(self.position_predictor(shared_features)) * self.image_size
#         angles = torch.tanh(self.orientation_predictor(shared_features))  # [-1, 1]
        
#         # Store current angles for loss computation
#         self.current_angles = angles
        
#         return self.generate_binary_bar(centers, angles)

#     def loss_function(self, pred, target):
#         # Get target orientation
#         target_angle = self.get_target_orientation(target)
        
#         # Use stored angles from forward pass
#         pred_angle = self.current_angles.squeeze(-1)
        
#         # Basic shape loss
#         pred = torch.clamp(pred, 0, 1)
#         binary_target = ((target != 116.0).float()).clamp(0, 1)
#         basic_loss = F.binary_cross_entropy(pred, binary_target)
        
#         # Orientation matching using sine and cosine
#         angle_loss = (
#             F.mse_loss(torch.cos(pred_angle * torch.pi), torch.cos(target_angle * torch.pi)) +
#             F.mse_loss(torch.sin(pred_angle * torch.pi), torch.sin(target_angle * torch.pi))
#         )
        
#         # Center region emphasis
#         center_region = pred[:, 140:180, 140:180]
#         target_region = binary_target[:, 140:180, 140:180]
#         center_loss = F.binary_cross_entropy(center_region, target_region)
        
#         return basic_loss + 5.0 * angle_loss + 2.0 * center_loss
#         # End of Selection
