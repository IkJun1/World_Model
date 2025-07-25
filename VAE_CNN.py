import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_channel, latent_dim=256):
        super().__init__()
        self.input_channel = input_channel
        self.latent_dim = latent_dim

        self.encode_cnn = nn.Sequential( # input은 64*64 이미지, 그리고 이미지 정규화 시켜줘야함 0~1로 
            nn.Conv2d(input_channel, 32, kernel_size=4, stride=2, padding=1), # 64-> 32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.SiLU(),
        ) 

        self.mu_linear = nn.Linear(4096, latent_dim)
        self.var_linear = nn.Linear(4096, latent_dim)

        self.decode_linear = nn.Linear(latent_dim, 4096)

        self.decode_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, input_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input_data):
        input_vector = self.encode_cnn(input_data)
        input_vector = torch.flatten(input_vector, start_dim=1) 

        mu = self.mu_linear(input_vector)
        log_var = self.var_linear(input_vector)

        return mu, log_var   

    def reparameterization(self, mu, log_var):
        var = torch.exp(log_var)
        std = var**(1/2)
        eps = torch.randn_like(mu)
        return (mu + std*eps) #element wise
        
    def decode(self, latent_vector):
        v = self.decode_linear(latent_vector)
        v = v.view(-1, 256, 4, 4)
        generate_image = self.decode_cnn(v)
        return generate_image

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        latent_vector = self.reparameterization(mu, log_var)
        output = self.decode(latent_vector)
        return output, mu, log_var

def loss_function(input, output, mu, log_var):
    BCE = F.binary_cross_entropy(input, output, reduction='none')
    BCE = torch.sum(BCE, dim=[1, 2, 3])

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1)

    total_loss = BCE+KLD

    return torch.mean(total_loss)