import torch 
from torch import nn, optim
import numpy as np
from tqdm.auto import trange

class VariationalEncoder(nn.Module):
    """
    Encodes a tensor (N, C, 158, 158) representing a batch of 158*158 images 
    into a latent tensor z of shape (N, C, z_dim)  
    """
    def __init__(self, nc, nf, z_dim):
        super(VariationalEncoder, self).__init__()
        self.nf = nf
        self.z_dim = z_dim
        self.layers = nn.Sequential(
            # 158*158 1 channel to 79*79 64 channels
            nn.Conv2d(nc, nf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            # 79*79 64 channels to 40*40 128 channels
            nn.Conv2d(nf, nf*2, 3, 2, 1, bias = False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace = True),

            # 40*40 128 channels to 20*20 256 channels
            nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace = True),

            # 20*20 256 channels to 10*10 512 channels
            nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace = True),

            # 10*10 512 channels to 5*5 1 channel
            nn.Conv2d(nf*8, nc, 4, 2, 1, bias = False)
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.linear1 = nn.Linear(26, z_dim)
        self.linear2 = nn.Linear(26, z_dim)

        # Tracking some important parameters (kl for the loss, mu and std to visualize the latent space)
        self.kl = 0
        self.mu = 0
        self.std = 0

    def reparameterize(self, mu, std):
        epsilon = self.N.sample(mu.shape)
        z = mu + std*epsilon
        return z

    
    def forward(self, x, condition):
        # Sending input (N,1,158,158) into the model; N = batch size
        x = x.unsqueeze(1)
        condition = condition.unsqueeze(1)
        x = self.layers(x)
        x = torch.flatten(x, start_dim = 1) # (N, 25)
        x = torch.cat((x, condition), dim = 1) # (N, 26)
        
        # mu.shape and std.shape = (N, z_dim)
        self.mu = self.linear1(x) 
        logVar = self.linear2(x)
        self.std = torch.exp(logVar/2)
        
        z = self.reparameterize(self.mu, self.std)

        # Updating the kl term value
        self.kl = 1/2*(self.mu**2 + torch.exp(logVar) -logVar -1).sum()

        return torch.cat((z, condition), dim = 1)



class Decoder(nn.Module):
    """
    Decode a latent tensor z of shape (N, z_dim) 
    into a batch of 158*158 galaxy images (N, C, 158, 158)  
    """
    def __init__(self, nc, nf, z_dim):
        super(Decoder, self).__init__()
        self.nc = nc
        self.layers = nn.Sequential(
            # 5*5 1 channel to 10*10 512 channels
            nn.ConvTranspose2d(nc, nf*8, 4, 2, 1),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(True),

            # 10*10 512 channels to 20*20 256 channels
            nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(True),

            # 20*20 256 channels to 40*40 128 channels
            nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            # 40*40 128 channels to 79*79 64 channels
            nn.ConvTranspose2d(nf*2, nf, 3, 2, 1, bias = False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            # 79*79 64 channels to 158*158 1 channel 
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias = False),
            nn.Sigmoid()
        )

        self.linear1 = nn.Linear(z_dim + 1, 25)
    
    def forward(self, z):
        # z shape =  (N, z_dim + 1)
        z = self.linear1(z) # (N, 25)
        z = z.reshape((-1, self.nc, 5, 5))
        return self.layers(z).squeeze()
        
class VariationalAutoencoder(nn.Module):
    """
    To define while calling:
    - nc = number of channels
    - nf = size of the feature map
    - z_dim = size of the latent space 

    Input > x = (N, 158, 158), batch of galaxy images; condition = (N,), batch of the redshifts associated to the galaxy images  
    Output > x_pred = (N, 158, 158) 
    """
    def __init__(self, nc, nf, z_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(nc, nf, z_dim)
        self.decoder = Decoder(nc, nf, z_dim)

    def forward(self, x, condition):
        z = self.encoder(x, condition)
        return self.decoder(z)

    def train_time(self, train_loader, val_loader, epochs = 100, learning_rate = 1e-3, beta = 0.1):
        device = 'cuda' 
        
        optimizer = optim.Adam(self.parameters(), lr = learning_rate) 
        loss_fn = nn.MSELoss(reduction = 'sum')
        train_loss = []
        val_loss = []
        mse = []
        kl = []
        
        with trange(epochs) as pbar:
            for epoch in pbar:
                if epoch == 40:
                    optimizer = optim.Adam(self.parameters(), lr = 1e-3*learning_rate)
                for x, condition in train_loader:
                    x, condition = x.to(device), condition.to(device)
                    x_pred = self.forward(x, condition)
                    loss = loss_fn(x_pred, x) + beta*self.encoder.kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    mse.append(loss_fn(x_pred, x).item())
                    kl.append(self.encoder.kl.item())
                    pbar.set_description(f"Train loss: {loss.item():.2g}")

                with torch.no_grad():
                    for x, condition in val_loader:
                        x, condition = x.to(device), condition.to(device)
                        x_pred = self.forward(x, condition)
                        loss = loss_fn(x_pred, x) + beta*self.encoder.kl
                        val_loss.append(loss.item())

        return np.array(train_loss), np.array(val_loss), np.array(mse), np.array(kl)

    def traintab(self, train_loader, val_loader, epochs = 100, learning_rate = 1e-3, beta = 0.1):
        device = 'cuda' 
        
        optimizer = optim.Adam(self.parameters(), lr = learning_rate) 
        loss_fn = nn.MSELoss(reduction = 'sum')
        train_loss = []
        val_loss = []
        mse = []
        kl = []
        
        with trange(epochs) as pbar:
            for epoch in pbar:
                for x, condition in train_loader:
                    x, condition = x.to(device), condition.to(device)
                    x_pred = self.forward(x, condition)
                    loss = loss_fn(x_pred, x) + beta[epoch]*self.encoder.kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    mse.append(loss_fn(x_pred, x).item())
                    kl.append(self.encoder.kl.item())
                    pbar.set_description(f"Train loss: {loss.item():.2g}")

                with torch.no_grad():
                    for x, condition in val_loader:
                        x, condition = x.to(device), condition.to(device)
                        x_pred = self.forward(x, condition)
                        loss = loss_fn(x_pred, x) + beta[epoch]*self.encoder.kl
                        val_loss.append(loss.item())

        return np.array(train_loss), np.array(val_loss), np.array(mse), np.array(kl)