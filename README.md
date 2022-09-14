
# Topic
This project aims to generate galaxy images based on data provided by the Hubble Space Telescope (HST). To do so, we are implementing an unsupervised machine learning technique called a Variational Autoencoder (Vae) based on statistical Bayesian inference. The trained Vae model allows us to decode a random latent variable $z$ sampled from a normal distribution $\mathcal{N}(0;1)$ into a realistic galaxy image. For further information on the general Vae model, please read : [Auto-encoding Variational Bayes, _Kingma and Welling_, 2014](https://arxiv.org/abs/1312.6114?context=cs.LG).  

We used two datasets : 
- 47 955 galaxies from Hubble's famous Deep Field image (the images have   
128 $\times$ 128 pixels)
- 81 499 galaxies and their associated redshifts from the Cosmic Survey (the images have 158 $\times$ 158 pixels) 

For each dataset, we developped a $\beta$-Vae architecture. For galaxy image generation, we found that the reconstruction performance of our model is better with low value for $\beta$. In our case, we fixed this hyperparameter to $\beta = 0.1$ during the whole training process.  

Based on [Kihyuk Sohn's paper](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html), we even implemented a Conditional Vae on the second dataset with the redshifts of each galaxy. In the end, our conditional vae is able to generate galaxy structures for a specific redshift. We can even do an interpolation of the same galaxy structure for different redshifts.  

# Models architecture
In the folder `Models architecture`, you will find the details of the different models used.  

First, two disentangled Vae models (one per dataset) with almost the same architecture (just a few changes made to the convolutional layers arguments due to the image size difference for each dataset). The models can take as input either a value or an array for the hyperparameter $\beta$. It can be interesting if you want to make $\beta$ change during the training process (i.e. $\beta$ increasing over each epoch). After experimentation of different behaviour for $\beta$, we concluded that we obtain the best model by fixing this hyperparameter to $\beta = 0.1$. We observe the same thibg for the conditional Vaes. 

Then, three Conditional Vae models (for the second dataset):    
- `cvae`: new input created by concatenation of the redshifts to the galaxy images into a second channel which is fed to the CNN. Then, we concatenate the redshifts to the latent variable $z$ into a second channel. The final output is the reconstructed galaxy image.

![Cvae](https://github.com/EnceladeCandy/vae-project/blob/main/cvae.png)

- `cvae2`: concatenation of the redshifts to the output of the encoder's CNN and to the latent variable $z$ before decoding. 

![Cvae 2](https://github.com/EnceladeCandy/vae-project/blob/main/cvae2.png)


- `fancy_cvae`: similar to `cvae` but the final output is a prediction of the galaxy images and the redshifts.  

![Fancy cvae](https://github.com/EnceladeCandy/vae-project/blob/main/fancy_cvae.png)

The performance of these architectures is really similar. The unique noteworthy difference is the training time which is shorter by 20 sec/epoch for `cvae2` compared to the others, so I would recommend using the `cvae2` architecture for conditioned galaxy image generation.  


# Notebooks
In the folder `notebooks`, you will find all the code related to each model's training and the evaluation of its performance:  
- Loss
- Image reconstruction 
- Image generation
- Latent space visualization 

# What's next ? 
## Normalizing flows
 Currently, we are generating images by giving some random samples of a Gaussian $\mathcal{N}(0,I)$ to the decoder. Since we are working with a disentangled conditional vae, we don't really know if these samples give us a good representation of our latent space. In fact, they are very unlikely to give us a good representation when $\beta$ is close to 0. We can visualize the latent space with 2d histograms to see if it's the case but it would be great to have a method to be sure we sample directly from the latent space. That's why the next step would be to implement a Normalizing flow to learn an invertible transformation from the rather "complex" probability distribution learnt by the vae to a simple Normal distribution. This would help the decoder during the generative process.

## Bigger dataset 
Training our model on a bigger dataset (314 000 galaxy images instead of 81 500).

## Fine-tuning of the model
We could maybe improve the results with some fine-tuning of the hyperparameters of the model or with different machine learning approaches that are not implemented yet (i.e. learning rate decay). 







