import torch.nn as nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F

from model.inner.st_module.net_module import Conv1d


class ObservationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(ObservationModel, self).__init__()
        # dense observation model
        self.num_hidden_layers = num_hidden_layers

        self.dec_in = nn.Conv1d(input_dim, hidden_dim, 1)
        self.dec_hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1) for _ in range(num_hidden_layers)])
        self.dec_out_1 = nn.Conv1d(hidden_dim, output_dim, 1)
        self.dec_out_2 = nn.Conv1d(hidden_dim, output_dim, 1)

        diffusion_embedding_dim = 128
        self.diffusion_projection = nn.Sequential(nn.Linear(diffusion_embedding_dim, hidden_dim), nn.Linear(hidden_dim, input_dim))

        # nn.init.xavier_uniform_(self.diffusion_projection.weight)

    def decode(self, z):
        b, num_m, _, t = z.shape
        z = z.view([b * num_m, -1, t])
        h = torch.relu(self.dec_in(z))
        for i in range(self.num_hidden_layers):
            #h = torch.relu(self.dec_hidden[i](h))
            h = self.dec_hidden[i](h)
        return self.dec_out_1(h).view([b, num_m, -1, t]), self.dec_out_2(h).view([b, num_m, -1, t])

    def forward(self, z, diffusion_emb=None):
        """
        likelihood function
        Args:
            z: list
        Returns:
            distributions of target_y
        """
        z = torch.cat(z, dim=2)
        if diffusion_emb is not None:
            diffusion_emb = self.diffusion_projection(diffusion_emb.squeeze(1).squeeze(-1))  # (B,channel)
            z = z + diffusion_emb.unsqueeze(1).unsqueeze(-1)
        mu, log_sigma = self.decode(z)
        return mu, 0.1 + 0.9 * F.softplus(log_sigma)
        #return mu