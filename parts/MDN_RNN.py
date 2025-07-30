import torch
from torch import nn
from torch import distributions
import numpy as np

class MDN(nn.Module):
    def __init__(self, input_size, num_dist, latent_size, temperature): # Need to add controller parameters
        super().__init__()
        self.input_size = input_size # It is the output vector size of the rnn
        self.num_dist = num_dist
        self.latent_size = latent_size
        self.tau = temperature

        self.phi_linear = nn.Linear(input_size, num_dist)
        self.mu_linear = nn.Linear(input_size, num_dist*latent_size)
        self.sigma_linear = nn.Linear(input_size, num_dist*latent_size)

        self.softmax = nn.Softmax(-1)

    def forward(self, input_data):
        mu = self.mu_linear(input_data)
        mu = mu.view(-1, self.num_dist, self.latent_size)

        sigma_log = self.sigma_linear(input_data)
        sigma = torch.exp(sigma_log) # Ensure positivity through the exponential function
        sigma = sigma.view(-1, self.num_dist, self.latent_size)

        phi = self.phi_linear(input_data)
        phi = self.softmax(phi/self.tau)

        return mu, sigma, phi 

class MDN_RNN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, latent_space_size=512, num_dist=3, temperature=1): # latent_space_size is fixed at 512. Do not change this parameter
        super().__init__()
        self.input_size = input_size
        self.num_dist = num_dist
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size+action_size, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.mdn = MDN(input_size=hidden_size, num_dist=num_dist, latent_size=latent_space_size, temperature=temperature)
    
    def sampling(self, mu, sigma, phi):
        mixture_distribution = distributions.Categorical(probs=phi)
        component_distribution = distributions.Normal(loc=mu, scale=sigma)
        mixture_gaussian = distributions.MixtureSameFamily(mixture_distribution, component_distribution)

        return mixture_gaussian.sample()

    def forward(self, input_z_vector, a_t_onehot):
        vector_sequence = torch.cat([input_z_vector, a_t_onehot], dim=2)
        output, _ = self.lstm(vector_sequence)

        mu, sigma, phi = self.mdn(output)

        z_t1 = self.sampling(mu, sigma, phi)

        return z_t1