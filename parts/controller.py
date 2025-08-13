import torch
from torch import nn

class controller(nn.Module):
    def __init__(self, action_size, z_vector_size, h_vector_size, hidden_size):
        super().__init__()
        self.action_size = action_size
        self.z_vector_size = z_vector_size # latent_space of VAE
        self.h_vector_size = h_vector_size # hidden_state of RNN

        self.input_linear = nn.Linear(z_vector_size+h_vector_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, action_size)

        self.softmax = nn.Softmax(-1)

    def forward(self, z_vector, h_vector):
        input_vector = torch.cat([z_vector, h_vector], dim=-1)
        hidden_vector = self.input_linear(input_vector)
        output = self.output_linear(hidden_vector)
        output_softmax = self.softmax(output)

        return output_softmax

def choice_control(p):
    action_indices = torch.multinomial(p, num_samples=1)
    return action_indices