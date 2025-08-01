import torch
from torch import nn
from torch import distributions
from torch.utils.data import Dataset
import numpy as np

class MDN(nn.Module):
    def __init__(self, input_size, num_dist, latent_size, temperature):
        super().__init__()
        self.input_size = input_size # It is the output vector size of the RNN
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
    def __init__(self, input_size, action_size, hidden_size=256, latent_space_size=256, num_dist=3, temperature=1): # input_size is latent_size of VAE
        super().__init__()
        self.input_size = input_size
        self.num_dist = num_dist # number of gaussian distrunutions in the GMM
        self.action_size = action_size
        self.hidden_size = hidden_size # hidden state size of lstm

        self.lstm = nn.LSTM(input_size=input_size+action_size, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.mdn = MDN(input_size=hidden_size, num_dist=num_dist, latent_size=latent_space_size, temperature=temperature)

    def forward(self, input_z_vector, a_t_onehot):
        vector_sequence = torch.cat([input_z_vector, a_t_onehot], dim=2)
        output, _ = self.lstm(vector_sequence)
        B, Seq, hidden_size = output.size()
        output = output.view(-1, hidden_size) # predict at hidden states of all sequences

        mu, sigma, phi = self.mdn(output)
        mu, sigma, phi = mu.view(B, Seq, self.num_dist, -1), sigma.view(B, Seq, self.num_dist, -1), phi.view(B, Seq, -1)

        return mu, sigma, phi
    
def sampling(mu, sigma, phi):
    mixture_distribution = distributions.Categorical(probs=phi)
    component_distribution = distributions.Normal(loc=mu, scale=sigma)
    mixture_gaussian = distributions.MixtureSameFamily(mixture_distribution, component_distribution)

    return mixture_gaussian.sample()

def mdn_rnn_loss(mu, sigma, phi, target):
    dist = distributions.Normal(loc=mu, scale=sigma)
    target = target.unsqueeze(1)
    log_prob = dist.log_prob(target)
    joint_log_prob = log_prob.sum(dim=-1)
    log_phi = torch.log(phi) 

    weighted_log_prob = log_phi + joint_log_prob # originally prob * phi, but log_prob + log_phi since we're in log-space
    log_likelihood = torch.logsumexp(weighted_log_prob, dim=-1)

    return -log_likelihood.mean()

class SequenceDataset(Dataset):
    def __init__(self, image_dataset, transforms, action_dataset, sequence_length=1000):
        self.images = image_dataset
        self.actions = action_dataset
        self.length = sequence_length
        self.transforms = transforms

    def __len__(self):
        return len(self.images)-self.length
    
    def __getitem__(self, idx):
        images = self.images[idx: idx+self.length]
        images_transform = [self.transforms(image) for image in images]
        images_tensor = torch.stack(images_transform)
        actions = self.actions[idx: idx+self.length]

        return images_tensor, torch.tensor(actions, dtype=torch.float32)