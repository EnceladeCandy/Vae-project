
# Topic
This project aims to generate galaxy images based on data provided by the Hubble Space Telescope (HST). To do so, we are implementing an unsupervised machine learning technique called a Variational Autoencoder (Vae) based on statistical Bayesian inference. The trained Vae model allows us to decode a random latent variable $z$ sampled from a normal distribution $\mathcal{N}(0;1)$ into a realistic galaxy image. For further information on the general Vae model, please read : [Auto-encoding Variational Bayes, _Kingma and Welling_, 2014](https://arxiv.org/abs/1312.6114?context=cs.LG).  

We used two datasets for which we developped different Vae architectures, and even implemented a Conditional Vae for the second one based on [Kihyuk Sohn's paper](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) : 
- 47 955 galaxies from Hubble's famous Deep Field image (the images have 128*128 pixels)
- 81 499 galaxies and their associated redshifts from the Cosmic Survey (the images have 158*158)  

# Models architecture
In the folder `Models architecture`, you will find the details of the different models used.  

First, two Vae models (one per dataset) with almost the same architecture (just a few changes made to the convolutional layers arguments due to the image size difference for each dataset).

Then, three Cvae models (for the second dataset):    
- `cvae`: new input created by concatenation of the redshifts to the galaxy images into a second channel which is fed to the CNN. Then, we concatenate the redshifts to the latent variable $z$ into a second channel. The final output is the reconstructed galaxy image.

![Cvae](https://github.com/EnceladeCandy/vae-project/blob/main/cvae.png)

- `cvae2`: concatenation of the redshifts to the output of the encoder's CNN and to the latent variable $z$ before decoding. 

![Cvae 2](https://github.com/EnceladeCandy/vae-project/blob/main/cvae2.png)


- `fancy_cvae`: similar to `cvae` but the final output is a prediction of the galaxy images and the redshifts.  

![Fancy cvae](https://github.com/EnceladeCandy/vae-project/blob/main/fancy_cvae.png)

The performance of these architectures is really similar. The unique noteworthy difference is the training time which is shorter by 20 sec/epoch for `cvae2` compared to the others, so I'd recommend using the `cvae2` architecture for conditionned generation of galaxy images. 


# Notebooks
In the folder `notebooks`, you will find all the code related to each model's training and the evaluation of its performance:  
- Loss
- Image reconstruction 
- Image generation
- Latent space visualization 

# References 





