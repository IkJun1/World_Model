import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_space_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_space_dim = latent_space_dim
        self.encode_linear = nn.Linear(input_dim, latent_space_dim)
        self.mu_linear = nn.Linear(latent_space_dim, latent_space_dim)
        self.var_linear = nn.Linear(latent_space_dim, latent_space_dim)
        self.decode_linear = nn.Linear(latent_space_dim, input_dim)

    def encode(self, input_vector):
        mu = self.mu_linear(input_vector)
        log_var = self.var_linear(input_vector)
        return mu, log_var
    
    def reparameterization(self, mu, log_var):
        var = torch.exp(log_var)
        std = var**(1/2)
        eps = torch.randn(mu.shape[0], self.latent_space_dim)
        return (mu + std*eps) #element wise
        
    def decode(self, latent_vector):
        generate_image = self.decode_linear(latent_vector)
        return generate_image

    def forward(self, input_data):
        v1 = self.encode_linear(input_data)
        mu, log_var = self.encode(v1)
        latent_vector = self.reparameterization(mu, log_var)
        output = self.decode(latent_vector)
        return output, mu, log_var

def loss_function(input, output, mu, log_var):
    MSE = F.mse_loss(input, output)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var))
    return MSE+KLD

class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image