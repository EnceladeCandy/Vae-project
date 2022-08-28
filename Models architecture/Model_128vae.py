import torch 
from torch import nn, optim
import numpy as np
from tqdm.auto import trange 

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        nc = 1
        nf = 64
        self.layers = nn.Sequential(
            # 128*128 images 1 channel to 64*64 images with 64 channels 
            nn.Conv2d(nc, nf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 64*64 images 64 channels to 32*32 images 2*64 channels
            nn.Conv2d(nf, nf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 32*32 images 128 channels to 16*16 images 4*64 channels
            nn.Conv2d(nf*2,nf*4, 4,2,1, bias = False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2,inplace = True),
            
            # 16*16 images 256 channels to 8*8 images 8*64 channels 
            nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace = True),
            
            #8*8 images 512 channels to 4*4 images 1 channel
            nn.Conv2d(nf*8, nc, 5, 1, 0, bias=False)
        )
        # From 4*4 to z dimension 
        self.linear1 = nn.Linear(nf//4, z_dim)
        self.linear2 = nn.Linear(nf//4, z_dim)
        
        # Distributions on gpu
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        
        # Tracking some important values
        self.kl = 0 
        self.mu = 0
        self.var = 0
        
    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        epsilon = self.N.sample(mu.shape)
        z = mu + std * epsilon
        return z
    
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim = 2)
        
        self.mu = self.linear1(x) # (batchsize, Channels, 16) >> (batchsize, Channels, z_dim)
        logVar = self.linear2(x)
        self.var = torch.exp(logVar)
        
        self.kl =1/2*(self.mu**2 + torch.exp(logVar)-logVar-1).sum()
        z = self.reparameterize(self.mu, logVar)
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        nc = 1
        nf = 64

        self.layers = nn.Sequential(
            # 4*4 images 1 channel to 8*8images 512 (8*64) channels 
            nn.ConvTranspose2d(nc, nf*8, 5,1,0, bias = False),
            nn.BatchNorm2d(nf*8),
            nn.ReLU(True), 
            
            # 8*8 images 512 channels to 16*16 images 256 channels 
            nn.ConvTranspose2d(nf*8,nf*4,4,2,1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.ReLU(True),
            
            # 16*16 images 256 channels to 32*32 images 128 channels
            nn.ConvTranspose2d(nf*4, nf*2, 4, 2,1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),
            
            # 32*32 images 128 channels to 64*64 images 64 channels
            nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            
            # 64*64 images 64 channels to 128*128 images 1 channel
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid() # to have pixels between 0 and 1
        )
        self.linear1 = nn.Linear(z_dim, nf//4)
        
    def forward(self, z):
        z = self.linear1(z)
        # -1 enables to adapt the size of z according to a specific batch size
        z = z.reshape((-1,1,4,4)) 
        return self.layers(z)


class VariationalAutoencoder(nn.Module):
    """
    To define before calling:
    - nc = number of channels of the input image (here=1)

    To define while calling:
    - nf = size of the feature map
    - z_dim = size of the latent space 
    """
    def __init__(self, z_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def train_time(self, train_loader, val_loader, epochs = 100, learning_rate = 1e-3, beta = 0.1):
        device = 'cuda' 
        
        # Making sure beta is a table of values 
        if type(beta) == float or type(beta) == int:
            beta = beta * torch.ones(epochs)
        
        optimizer = optim.Adam(self.parameters(), lr = learning_rate) 
        loss_fn = nn.MSELoss(reduction = 'sum')
        train_loss = []
        val_loss = []
        mse = []
        kl = []
        
        with trange(epochs) as pbar:
            for epoch in pbar:
                for x in train_loader:
                    x = x.unsqueeze(1).to(device) # (N, 1, 158, 158) on gpu
                    x_pred = self.forward(x)
                    loss = loss_fn(x_pred, x) + beta[epoch]*self.encoder.kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    mse.append(loss_fn(x_pred, x).item())
                    kl.append(self.encoder.kl.item())
                    pbar.set_description(f"Train loss: {loss.item():.2g}")
        
                with torch.no_grad():
                    for x in val_loader:
                        x = x.unsqueeze(1).to(device) # (N, 1, 158, 158) on gpu
                        x_pred = self.forward(x)
                        loss = loss_fn(x_pred, x) + beta[epoch]*self.encoder.kl
                        val_loss.append(loss.item())

        return np.array(train_loss), np.array(val_loss), np.array(mse), np.array(kl)