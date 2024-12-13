import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import warnings
from pathlib import Path

class Trainer:
    '''
    Class consisting of methods to train an artificial neural network (ANN).

    Inputs:
    -------
        n_inputs: Scalar
            Number of input channels
        n_pixels: Scalar
            Number of pixels of the target image, flattened to a vector
        n_steps: Scalar
            Number of simulation steps (not really used in standard ANN)
        n_epochs: Scalar
            Number of epochs
        device: String
            Either 'cpu' or 'cuda:0'
        model_save_dir: String
            Directory where the model should be saved at.
        net_name: String
            Name of the network to train, either 'ANN'. Default is 'ANN'.
        lr: Scalar
            Learning rate for the Adam optimizer, default is 1e-4.

    '''
    def __init__(self, n_inputs, n_pixels, n_epochs, n_steps, device, model_save_dir, net_name='ANN-CNN', lr=1e-4):
        self.n_inputs = n_inputs
        self.n_pixels = n_pixels
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.lr = lr
        
        self.model_save_dir = model_save_dir
        self.net_name = net_name
        self.device = device
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _choose_network(self):
        '''
        Choose which network to train and define or call it.

        Output:
        -------
            network: torch.nn class
                Neural network class definition
        '''
        if self.net_name == 'ANN-CNN':
            network = nn.Sequential(OrderedDict([
            ('v1_simple', nn.Conv2d(in_channels=self.n_inputs, out_channels=16, kernel_size=5, stride=1, padding=1)),  # First Conv Layer
            ('activation1', nn.ReLU()),  # Activation
            ('v1_complex', nn.MaxPool2d(kernel_size=2, stride=2)),  # Pooling Layer
            ('v2', nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=3)),  # Second Conv Layer
            ('activation2', nn.ReLU()),  # Activation
            ('v2_complex', nn.AdaptiveMaxPool2d(1)),  # Adaptive Pooling Layer
            ('flat', nn.Flatten()),  # Flatten the output for fully connected layer
            ('fc1', nn.Linear(8, self.n_pixels)),  # Fully connected layer
            ('activation3', nn.ReLU())  # Activation for the fully connected layer
        ]))
            network = network.to(self.device)
            # Initialize weights for the network (if necessary)
            network.apply(self._init_weights)

            # Freeze all layers except for the last layer (for fine-tuning)
            for name, param in network.named_parameters():
                if 'fc1' not in name:  # Exclude the last fully connected layer
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # Keep the last fully connected layer trainable

        else:
            warnings.warn("The 'net_name' parameter only accepts 'ANN-CNN' so far.")
            return None

        return network

    def _forward(self, network, data):
        '''
        Forward pass through the network.

        Inputs:
        -------
            network: torch.nn class
                Neural network
            data: (n_batch, n_channels, n_pix, n_pix) Torch tensor
                Input data in the correct shape

        Outputs:
        --------
            output: (n_batch, n_pixels) Torch tensor
                Output of the network
        '''
        return network(data)

    def train(self, spk_in, target):
        '''
        Train the selected network.

        Inputs:
        ------
            spk_in: (n_batch, n_channels, n_pix, n_pix) Torch tensor
                Input image tensor.
            target: (n_batch, n_pixels) Torch tensor
                Target image, where n_pixels = n_pix x n_pix

        Outputs:
        --------
            loss_hist: (n_epochs, ) Torch tensor
                Loss over epochs
            decoded_image: (n_batch, n_pixels) Torch tensor
                Decoded image (output of the network)
            network: torch.nn class
                Trained network
        '''
        # Initialize network
        network = self._choose_network()

        # Gradient descent optimizer
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()

        loss_hist = torch.zeros(self.n_epochs)
        
        network.train()

        # Training loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()  
            data = spk_in.to(self.device)
            target = target.to(self.device)

            # Print the shape of the input and target tensors
            print(f"Epoch {epoch + 1}:")
            print(f"Input (spk_in) shape: {data.shape}")
            print(f"Target shape: {target.shape}")

            # Forward pass
            output = self._forward(network, data)

            # Print the shape of the output tensor
            print(f"Output shape before reshaping: {output.shape}")

            output = output.view(-1, 320, 320)

            print(f"Output shape after reshaping: {output.shape}")

            # Compute loss
            loss = mse_loss(output, target.float()) 
            loss_hist[epoch] = loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Save the trained model
        torch.save(network.state_dict(), Path(self.model_save_dir) / 'ann_model.pt')

        return loss_hist, output, network

    def eval(self, network, spk_in, target):
        '''
        Evaluate the selected network (no gradients are computed).

        Inputs:
        ------
            network: torch.nn class
                Neural network
            spk_in: (n_batch, n_channels, n_pix, n_pix) Torch tensor
                Input image tensor.
            target: (n_batch, n_pixels) Torch tensor
                Target image

        Outputs:
        --------
            loss: Torch tensor
                Validation loss
            decoded_image: (n_batch, n_pixels) Torch tensor
                Decoded image
        '''
        network.eval()
        mse_loss = nn.MSELoss()

        with torch.no_grad():
            data = spk_in.to(self.device)
            target = target.to(self.device)

            # Forward pass
            output = self._forward(network, data)

            output = output.view(-1, 320, 320)

            # Compute loss
            loss = mse_loss(output, target.float()) 

            #Get decoded image
            decoded_image = output.clone()

        return loss.item(), decoded_image
