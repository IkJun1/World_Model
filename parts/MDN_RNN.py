import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        self.reward_linear = nn.Linear(hidden_size, 1)
        self.done_linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_z_vector, a_t_onehot, pre_hidden=None, length=None):
        max_seq_len = input_z_vector.size(1)
        
        vector_sequence = torch.cat([input_z_vector, a_t_onehot], dim=2)

        if length is not None:
            vector_sequence = pack_padded_sequence(vector_sequence, length, batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(vector_sequence, pre_hidden)

        if length is not None:
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=max_seq_len)

        B, Seq, hidden_size = output.size()
        output = output.reshape(-1, hidden_size) # predict at hidden states of all sequences

        mu, sigma, phi = self.mdn(output)
        mu, sigma, phi = mu.view(B, Seq, self.num_dist, -1), sigma.view(B, Seq, self.num_dist, -1), phi.view(B, Seq, -1)

        reward = self.reward_linear(output)
        reward = reward.view(B, Seq)

        done = self.done_linear(output)
        done = self.sigmoid(done)
        done = done.view(B, Seq)

        return mu, sigma, phi, reward, done, output, (h_n, c_n)
    
def sampling(mu, sigma, phi):
    mixture_distribution = distributions.Categorical(probs=phi)
    component_distribution = distributions.Normal(loc=mu, scale=sigma)
    component_dist = distributions.Independent(
        component_distribution,
        reinterpreted_batch_ndims=1
    )
    mixture_gaussian = distributions.MixtureSameFamily(mixture_distribution, component_dist)

    return mixture_gaussian.sample()

def mdn_rnn_loss(mu, sigma, phi, target, p_reward, t_reward, p_done, t_done, mask=None, reward_weights=1): # p_reward is predicted reward by the model, t_reward is target reward
    dist = distributions.Normal(loc=mu, scale=sigma) # this dimension is have to same of the target dimension
    target = target.unsqueeze(1)
    log_prob = dist.log_prob(target)
    joint_log_prob = log_prob.sum(dim=-1)
    log_phi = torch.log(phi) 

    weighted_log_prob = log_phi + joint_log_prob # originally prob * phi, but log_prob + log_phi since we're in log-space
    log_likelihood = torch.logsumexp(weighted_log_prob, dim=-1)

    mse = F.mse_loss(p_reward, t_reward, reduction='none')

    done_bce = F.binary_cross_entropy(p_done, t_done, reduction='none')

    if mask is not None:
        mse = mse*mask
        log_likelihood = log_likelihood*mask

        n_mask = mask.sum()

        mean_mse = mse.sum()/n_mask
        mean_log_likelihood = log_likelihood.sum()/n_mask
        done_bce = done_bce.sum()/n_mask

        return -mean_log_likelihood + reward_weights*mean_mse + done_bce

    return -log_likelihood.mean() + reward_weights*mse.mean() + done_bce.mean()

class SequenceDataset(Dataset):
    def __init__(self, image_dataset, transforms, action_dataset, reward_dataset, episodes):
        # self.H, self.W, self. C = image_dataset[0].shape
        self.images = image_dataset
        self.actions = action_dataset
        self.rewards = reward_dataset
        self.episodes = episodes

        self.epi_length = [episodes[i+1]-episodes[i] for i in range(len(episodes)-1)]
        self.max_length = max(self.epi_length)

        self.transforms = transforms

    def __len__(self):
        return len(self.epi_length)-2 # not use the last sequence (because incomplete)
    
    def __getitem__(self, idx):
        image_seq = self.images[self.episodes[idx]: self.episodes[idx+1]]
        images_padded = np.pad(image_seq, ((0, self.max_length-self.epi_length[idx]), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        images_transform = [self.transforms(image) for image in images_padded]
        images_tensor = torch.stack(images_transform)

        action_seq = self.actions[self.episodes[idx]: self.episodes[idx+1]]
        actions_padded = np.pad(action_seq, ((0, self.max_length-self.epi_length[idx]), (0, 0)), mode='constant', constant_values=0)

        reward_seq = self.rewards[self.episodes[idx]: self.episodes[idx+1]]
        rewards_padded = np.pad(reward_seq, (0, self.max_length-self.epi_length[idx]), mode='constant', constant_values=0)

        seq_length = self.epi_length[idx]-1

        mask = torch.zeros(self.max_length-1)
        mask[:seq_length+1] = 1

        done = torch.zeros(self.max_length)
        done[self.epi_length[idx]-1] = 1

        return images_tensor, torch.tensor(actions_padded, dtype=torch.float32), torch.tensor(rewards_padded, dtype=torch.float32), torch.tensor(seq_length, dtype=torch.int64), mask, done