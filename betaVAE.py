import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class VAELoss(nn.Module):
    def __init__(self, beta=0.3):
        super().__init__()
        self.beta = beta  # Weight of the KL divergence term

    def forward(self, x_hat, x, mu, logvar, z_sampled, task_ids=None):
        """
        ELBO loss
        """
        mse_loss = F.mse_loss(x, x_hat)
        kl_loss = (torch.sum(-0.5 - 0.5 * logvar + 0.5 * mu.pow(2) + 0.5 * logvar.exp(), dim=1)).mean()

        loss = mse_loss + self.beta * kl_loss

        return loss
    
class VAELossVarBeta(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x, mu, logvar, z_sampled, beta, task_ids=None):
        """
        ELBO loss with Beta (weight of KL divergence term) in forward call to enable beta-scheduling
        """
        mse_loss = F.mse_loss(x, x_hat)
        kl_loss = (torch.sum(-0.5 - 0.5 * logvar + 0.5 * mu.pow(2) + 0.5 * logvar.exp(), dim=1)).mean()

        loss = mse_loss + beta * kl_loss

        return loss


class StateVariationalEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 96, 96)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1),   # 96->48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),      # 48->24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),     # 24->12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),    # 12->6
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),    # 6->3
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear_input_dim = 3 * 3 * 512
        self.fc1 = nn.Linear(self.linear_input_dim, 512)
        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        input_shape = x.shape
        x = x.view(-1, self.num_channels, 96, 96) # Input image parameters
        x = self.encoder(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        mu = mu.view(*input_shape[:-3], self.latent_dim)
        log_var = log_var.view(*input_shape[:-3], self.latent_dim)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick to sample from N(mu, std) from N(0,1)
        :param mu: <torch.Tensor> of shape (..., latent_dim)
        :param logvar: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """

        stdev = torch.exp(0.5 * logvar)
        e = torch.randn_like(stdev)
        sampled_latent_state = mu + e * stdev

        return sampled_latent_state


class StateDecoder(nn.Module):
    """
    Reconstructs the state from a latent space.
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 3 * 3 * 512)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 3->6
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6->12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 12->24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 24->48
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=2, padding=1),  # 48->96
            nn.Sigmoid()  # Images are scaled [0,1]
        )

    def forward(self, z):
        input_shape = z.shape
        z = z.view(-1, self.latent_dim)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 512, 3, 3)
        x = self.decoder(x)
        return x.view(*input_shape[:-1], self.num_channels, 96, 96)


class StateVAE(nn.Module):
    """
    State AutoEncoder
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = StateVariationalEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 96, 96)
        :return:
            reconstructed_state: <torch.Tensor> of shape (..., num_channels, 96, 96)
            mu: <torch.Tensor> of shape (..., latent_dim)
            log_var: <torch.Tensor> of shape (..., latent_dim)
            latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        mu, log_var = self.encoder(state)  # mean and log variance obtained from encoding state
        latent_state = self.encoder.reparameterize(mu, log_var)  # sample from the latent space feeded to the decoder
        reconstructed_state = self.decoder(latent_state)  # decoded states from the latent_state
        
        return reconstructed_state, mu, log_var, latent_state

    def encode(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 96, 96)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        mu, logvar = self.encoder(state)
        latent_state = self.encoder.reparameterize(mu, logvar)

        return latent_state

    def decode(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., num_channels, 96, 96)
        """
        reconstructed_state = self.decoder(latent_state)
        return reconstructed_state

    def reparameterize(self, mu, logvar):
        return self.encoder.reparameterize(mu, logvar)
