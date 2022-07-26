import matplotlib.pyplot as plt
import torch
from zmq import device



##################################################### Reconstruction #############################################################################

class reconstruct():
    def __init__(self, autoencoder, hyperparameters):
        self.Vae = autoencoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batchsize = hyperparameters["batch size"] 

    def cvae(self, galaxies, zphot, colour = 'hot', num_images = 10, title = False, img_title = 'Reconstruction Cvae'):
       # Input of cvae and cvae 2 is galaxies + redshifts
        galaxies_pred = self.Vae(galaxies.to(self.device), zphot.to(self.device))

        figure, axs = plt.subplots(2, num_images, figsize = (num_images, 2))
        for j in range(num_images): 
            rdm_label = torch.randint(self.batchsize, size =(1,))
            axs[0,j].imshow(galaxies[rdm_label].squeeze(), cmap = colour)
            axs[0,j].axis("off")
            axs[0,j].annotate(f"{zphot[rdm_label].item():.2f}", xycoords = "axes fraction", xy = (0.7, 0.85), color = 'white', fontsize = 7)
            axs[1,j].imshow(galaxies_pred[rdm_label].detach().to('cpu').numpy().squeeze(), cmap = colour)
            axs[1,j].axis("off")
        
        if title:
            figure.suptitle(img_title, y = 0.22)
        
        plt.subplots_adjust(wspace= 0.1 , hspace= 0.1 )


    def fancy_cvae(self, galaxies, zphot, colour = 'hot', num_images = 10, title = False, img_title = 'Reconstruction fancy cvae'):
        galaxies_pred, zphot_pred = self.Vae(galaxies.to(self.device), zphot.to(self.device))
        
        figure, axs = plt.subplots(2, num_images, figsize = (num_images, 2))
        
        for j in range(num_images): 
            rdm_label = torch.randint(self.batchsize, size =(1,))
            axs[0,j].imshow(galaxies[rdm_label].squeeze(), cmap = colour)
            axs[0,j].axis("off")
            axs[0,j].annotate(f"{zphot[rdm_label].item():.2f}", xycoords = "axes fraction", xy = (0.7, 0.85), color = 'white', fontsize = 6)
            
            axs[1,j].imshow(galaxies_pred[rdm_label].detach().to('cpu').numpy().squeeze(), cmap = colour)
            axs[1,j].axis("off")
            axs[1,j].annotate(f"{zphot_pred[rdm_label].item():.2f}", xycoords = "axes fraction", xy = (0.7, 0.85), color = 'white', fontsize = 6)
        
        if title:
            figure.suptitle(title, y = 0.22)
        plt.subplots_adjust(wspace= 0.1 , hspace= 0.1 )

    
    def vae158(self, galaxies, redshift, colour = 'hot', num_images = 10, title = False, img_title = 'Reconstruction Vae'):
        # Input of cvae and cvae 2 is galaxies + redshifts
        galaxies_pred = self.Vae(galaxies.unsqueeze(1).to(self.device))

        figure, axs = plt.subplots(2, num_images, figsize = (num_images, 2))
        
        for j in range(num_images): 
            rdm_label = torch.randint(self.batchsize, size =(1,))
            axs[0,j].imshow(galaxies[rdm_label].squeeze(), cmap = colour)
            axs[0,j].axis("off")
            axs[0,j].annotate(f"{redshift[rdm_label].item():.2f}", xycoords = "axes fraction", xy = (0.7, 0.85), color = 'white', fontsize = 6)
            
            axs[1,j].imshow(galaxies_pred[rdm_label].detach().to('cpu').numpy().squeeze(), cmap = colour)
            axs[1,j].axis("off")
        
        if title:
            figure.suptitle(img_title, y = 0.22)
        plt.subplots_adjust(wspace= 0.1 , hspace= 0.1)
    

    def vae128(self, galaxies, colour = 'hot', num_images = 10, title = False, img_title = 'Reconstruction Vae'):
        # Input of cvae and cvae 2 is galaxies + redshifts
        galaxies_pred = self.Vae(galaxies.unsqueeze(1).to(self.device))

        figure, axs = plt.subplots(2, num_images, figsize = (num_images, 2))
        
        for j in range(num_images): 
            rdm_label = torch.randint(self.batchsize, size =(1,))
            
            axs[0,j].imshow(galaxies[rdm_label].squeeze(), cmap = colour)
            axs[0,j].axis("off")

            axs[1,j].imshow(galaxies_pred[rdm_label].detach().to('cpu').numpy().squeeze(), cmap = colour)
            axs[1,j].axis("off")
        
        if title:
            figure.suptitle(img_title, y = 0.22)
        plt.subplots_adjust(wspace= 0.1 , hspace= 0.1)

####################################################################### Generation #############################################################

class generate():
    def __init__(self, autoencoder, hyperparameters):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Vae = autoencoder
        self.z_dim = hyperparameters["z_dim"]


    def cvae(self, redshift_value, rows = 8, cols = 8, colour = 'hot', title = False, figure_title = "Generation cvae architecture"):
        
        if type(redshift_value) == torch.Tensor():
            redshift = redshift_value.to(self.device)
        else: 
            redshift = torch.tensor([redshift_value]).to(self.device)

        figure, axs = plt.subplots(rows, cols, figsize = (cols, rows))

        for i in range(rows):
            for j in range(cols):
                z_generated = torch.normal(0, 1, size = (1, 1, self.z_dim)).to(self.device)
                pred_galaxies = self.Vae.decoder(self.Vae.concatenate(z_generated, redshift))
                
                axs[i,j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
                axs[i,j].axis('off')
                axs[i,j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)

        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)


    def cvae2(self, redshift_value, rows = 8, cols = 8, colour = 'hot', title = False, figure_title = "Generation cvae2 architecture"):
        
        if type(redshift_value) == torch.Tensor():
            redshift = redshift_value.to(self.device)
        else: 
            redshift = torch.tensor([redshift_value]).to(self.device)

        figure, axs = plt.subplots(rows, cols, figsize = (cols, rows))
        for i in range(rows):
            for j in range(cols):
                z_generated = torch.normal(0, 1, size = (1, self.z_dim)).to(self.device)
                z_cond = torch.cat((z_generated, redshift.unsqueeze(1)), dim = 1)
                pred_galaxies = self.Vae.decoder(z_cond)
                
                axs[i,j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
                axs[i,j].axis('off')
                axs[i,j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)

        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)


    def fancy_cvae(self, redshift_value, rows = 8, cols = 8, colour = 'hot', title = False, figure_title = "Generation fancy cvae architecture"):
        
        if type(redshift_value) == torch.Tensor():
            redshift = redshift_value.to(self.device)
        else: 
            redshift = torch.tensor([redshift_value]).to(self.device)

        figure, axs = plt.subplots(rows, cols, figsize = (cols, rows))
        for i in range(rows):
            for j in range(cols):
                z_generated = torch.normal(0, 1, size = (1, 1, self.z_dim)).to(self.device)
                pred_galaxies, pred_redshift = self.Vae.decoder(self.Vae.encoder.concatenate2(z_generated, redshift))
                
                axs[i,j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
                axs[i,j].axis('off')

                axs[i,j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)
                axs[i,j].annotate(f"{pred_redshift.item():.2f}",  xy = (0.7, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)
                
        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)

    def vae(self, rows = 8, cols = 8, colour = 'hot', title = False, figure_title = "Generation vae architecture"):
        
        figure, axs = plt.subplots(rows, cols, figsize = (cols, rows))
        for i in range(rows):
            for j in range(cols):
                z_generated = torch.normal(0, 1, size = (1, self.z_dim)).to(self.device)
                pred_galaxies = self.Vae.decoder(z_generated)
                
                axs[i,j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
                axs[i,j].axis('off')
                
        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)

###################################################################### Interpolation ##########################################################

class interpolate():
    def __init__(self, autoencoder, hyperparameters):
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.Vae = autoencoder
        self.z_dim = hyperparameters["z_dim"]


    def cvae(self, redshift_tensor, cols = 8, colour = 'hot', title = False, figure_title = "Generation cvae architecture"):
        
        figure, axs = plt.subplots(1, cols, figsize = (cols, 1))
        z_generated = torch.normal(0, 1, size = (1, 1, self.z_dim)).to(self.device)
    
        for j in range(cols):
            redshift = torch.tensor([redshift_tensor[j]]).to(self.device)
            pred_galaxies = self.Vae.decoder(self.Vae.concatenate(z_generated, redshift))
            
            axs[j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
            axs[j].axis('off')
            axs[j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)

        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)


    def cvae2(self, redshift_tensor, cols = 8, colour = 'hot', title = False, figure_title = "Generation cvae2 architecture"):
        
        figure, axs = plt.subplots(1, cols, figsize = (cols, 1))
        z_generated = torch.normal(0, 1, size = (1, self.z_dim)).to(self.device)
        
        for j in range(cols):
            redshift = torch.tensor([redshift_tensor[j]]).unsqueeze(0).to(self.device)
            z_cond = torch.cat((z_generated, redshift), dim = 1)
            pred_galaxies = self.Vae.decoder(z_cond)
            
            axs[j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
            axs[j].axis('off')
            axs[j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)

        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)


    def fancy_cvae(self, redshift_tensor, cols = 8, colour = 'hot', title = False, figure_title = "Interpolation fancy cvae architecture"):
        
        figure, axs = plt.subplots(1, cols, figsize = (cols, 1))
        z_generated = torch.normal(0, 1, size = (1, 1, self.z_dim)).to(self.device)

        for j in range(cols):
            redshift = torch.tensor([redshift_tensor[j]]).to(self.device)
            pred_galaxies, pred_redshift = self.Vae.decoder(self.Vae.encoder.concatenate2(z_generated, redshift))
            
            axs[j].imshow(pred_galaxies.detach().to('cpu').squeeze().numpy(), cmap = colour)
            axs[j].axis('off')

            axs[j].annotate(f"{redshift.item():.2f}",  xy = (0.1, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)
            axs[j].annotate(f"{pred_redshift.item():.2f}",  xy = (0.7, 0.85), xycoords = "axes fraction", color = "white", fontsize = 6)
                
        if title:
            figure.suptitle(figure_title, y = 0.12, fontsize = 10)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
