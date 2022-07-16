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
            nn.Conv2d(nf*8, 1, 4, 2, 1, bias = False)
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.linear1 = nn.Linear(25, z_dim)
        self.linear2 = nn.Linear(25, z_dim)

        # Tracking some important parameters (kl for the loss, mu and std to visualize the latent space)
        self.kl = 0
        self.mu = 0
        self.std = 0

    def reparameterize(self, mu, std):
        epsilon = self.N.sample(mu.shape)
        z = mu + std*epsilon
        return z

    def concatenate(self, x, condition):
        """
        Concatenate a tensor x of shape (N, k) and a condition tensor (N,) along the channel dimension
        Returns the concatenated tensor x_cond = (N, 2, k)  
        """
        condition_img = condition[:, None, None]*torch.ones_like(x)
        x_cond = torch.cat((x, condition_img), dim = 1)
        return x_cond


    def forward(self, x_cond):
        # x_cond = (N, 2, 158, 158) created during the training loop 
        condition = x_cond[:,1,0,0]
        x_cond = self.layers(x_cond) # (N, 1, 5, 5)
        x_cond = torch.flatten(x_cond, start_dim = 2)
        
        # mu.shape and std.shape = (N, 1, z_dim)
        self.mu = self.linear1(x_cond) 
        logVar = self.linear2(x_cond)
        self.std = torch.exp(logVar/2)
        
        z = self.reparameterize(self.mu, self.std) # (N, 1, z_dim)
        z_cond = self.concatenate(z, condition)

        # Updating the kl term value
        self.kl = 1/2*(self.mu**2 + torch.exp(logVar) -logVar -1).sum()

        return z_cond



class Decoder(nn.Module):
    """
    Decode a latent tensor z of shape (N, C, z_dim) 
    into a batch of 158*158 galaxy images (N, C, 158, 158)  
    """
    def __init__(self, nc, nf, z_dim):
        super(Decoder, self).__init__()
        self.nc = nc
        self.layers = nn.Sequential(
            # 5*5 2 channels to 10*10 512 channels
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

            # 79*79 64 channels to 158*158 2 channels
            nn.ConvTranspose2d(nf, 1, 4, 2, 1, bias = False),
            nn.Sigmoid()
        )
        self.linear1 = nn.Linear(z_dim, 25)

    def forward(self, z_cond):

        z_cond = self.linear1(z_cond).reshape((-1, self.nc, 5, 5))
        x_pred = self.layers(z_cond)
        return x_pred.squeeze()


class VariationalAutoencoder(nn.Module):
    """
    To define before calling:
    - nc = number of channels of the input image (here=1)

    To define while calling:
    - nf = size of the feature map
    - z_dim = size of the latent space 

    Reconstruct a image-based input
    """
    def __init__(self, nc, nf, z_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(nc, nf, z_dim)
        self.decoder = Decoder(nc, nf, z_dim)

    def forward(self, x_cond):
        z_cond = self.encoder(x_cond)
        return self.decoder(z_cond)
    
    def concatenate(self, x, condition):
        x = x.unsqueeze(1)
        condition_img = condition[:, None, None, None]*torch.ones_like(x)
        x_cond = torch.cat((x, condition_img), dim = 1)
        return x_cond
    

    def train_time(self, train_loader, epochs = 100, learning_rate = 1e-3, beta = 0.1):
        device = 'cuda' 
        
        # No tracking of the iotimizer during training maybe ? 
        optimizer = optim.Adam(self.parameters(), lr = learning_rate) 
        loss_fn = nn.MSELoss(reduction = 'sum')
        train_loss = []
        #val_loss = []
        mse = []
        kl = []
        
        with trange(epochs) as pbar:
            for epoch in pbar:
                for x, condition in train_loader:
                    x = x.to(device) # (N, 158, 158) on gpu
                    condition = condition.to(device)
                    x_cond = self.concatenate(x, condition)
                    x_pred = self.forward(x_cond)
                    loss = loss_fn(x_pred, x) + beta*self.encoder.kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    mse.append(loss_fn(x_pred, x).item())
                    kl.append(self.encoder.kl.item())
                    pbar.set_description(f"Train loss: {loss.item():.2g}")

                """
                with torch.no_grad():
                    for x in val_loader:
                        x = x.unsqueeze(1).to(device) # (N, 1, 158, 158) on gpu
                        x_pred = self.forward(x)
                        loss = loss_fn(x_pred, x) + beta*self.encoder.kl
                        val_loss.append(loss.item())
                """

        return np.array(train_loss), np.array(mse), np.array(kl)
