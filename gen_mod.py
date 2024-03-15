import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional
from scipy import interpolate
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
from scipy.signal import find_peaks
from torch.autograd import Variable
from scipy.stats import ks_2samp, chisquare
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

cuda = False #True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


##################################################
###                                            ###
###                   GAN                      ###
###                                            ###
##################################################


class Generator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=64, batch_norm=False):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(n_inputs, hidden_size),
                                   nn.Tanh())
        if batch_norm:
            self.model.append(nn.BatchNorm1d(hidden_size, 0.8))
        
        self.model.append(nn.Linear(hidden_size, n_outputs))

    def forward(self, x_noise, x_cond):
        x = torch.cat((x_noise, x_cond), dim=1)
        x = self.model(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, n_inputs, hidden_size=64, gan_type='V'):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(n_inputs, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1))
        
        if gan_type == 'V':
            self.model.append(nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        return x


##################################################
###                                            ###
###                 CPDGAN                     ###
###                                            ###
##################################################


# Gradient pealty for WGAN-GP
def compute_gp(discriminator, real_test_data, gen_data):

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_test_data.size(0), 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_test_data + ((1 - alpha) * gen_data)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = Variable(Tensor(real_test_data.size(0), 1).fill_(1.0), requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class CPDGAN(object):
    def __init__(self, gan_type='V', batch_size=32, n_epochs=10, n_disc=1, latent_dim=10, out_dim=1, hidden_dim=64, lambda_gp=1, batch_norm=False):
        
        self.gan_type   = gan_type
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.latent_dim = latent_dim
        self.n_disc     = n_disc
        self.lambda_gp  = lambda_gp
        self.gen_prior  = None
        
        self.generator     = Generator(self.latent_dim + 1, out_dim, hidden_dim, batch_norm)
        self.discriminator = Discriminator(out_dim, hidden_dim, gan_type)
        
        self.opt_gen  = torch.optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        self.loss_history_disc = []
        self.loss_history_gen  = []
        
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
    
    
    def fit(self, X, X_cond):
        
        X = Tensor(X)
        X_cond = Tensor(X_cond)
        X_train = TensorDataset(X, X_cond)
        
        if self.gan_type == 'V':
            # Loss function
            adversarial_loss = torch.nn.BCELoss()
        
        # Turn on training
        self.generator.train(True)
        self.discriminator.train(True)
        
        # Fit GAN
        for e in range (self.n_epochs):
            for ref_batch, cond_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True):
                
                if self.gan_type == 'V':
                    real = Variable(Tensor(ref_batch.size(0), 1).fill_(1.0), requires_grad=False)
                    fake = Variable(Tensor(ref_batch.size(0), 1).fill_(0.0), requires_grad=False)
                
                for _ in range(self.n_disc):

                    # Discriminator Training

                    noise_batch = Variable(Tensor(np.random.normal(0, 1, (len(ref_batch), self.latent_dim))))

                    gen_batch = self.generator(noise_batch, cond_batch)
                    
                    D_fake = self.discriminator(gen_batch.detach())
                    D_real = self.discriminator(ref_batch)

                    if self.gan_type == 'V':
                        real_loss = adversarial_loss(D_fake, fake)
                        fake_loss = adversarial_loss(D_real, real)
                        loss_disc = (real_loss + fake_loss) / 2
                    else:
                        loss_disc = torch.mean(D_fake) - torch.mean(D_real)
                        grad_p = compute_gp(self.discriminator, ref_batch, gen_batch.detach())
                        loss_disc += self.lambda_gp * grad_p

                    self.loss_history_disc.append(loss_disc.item())

                    self.opt_disc.zero_grad()
                    loss_disc.backward()
                    self.opt_disc.step()
                
                # Generator Training
                
                gen_batch = self.generator(noise_batch, cond_batch)
                
                D_fake = self.discriminator(gen_batch)
                
                if self.gan_type == 'V':
                    loss_gen = adversarial_loss(D_fake, real)
                else:
                    loss_gen = -torch.mean(D_fake)
                    
                self.loss_history_gen.append(loss_gen.item())
                
                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()
                
        # Turn off training
        self.generator.train(False)
        self.discriminator.train(False)
        
        
    def sample(self, n):
        noise = Variable(Tensor(np.random.normal(0, 1, (n, self.latent_dim))))
        X_gen = self.generator(noise)
        return X_gen.detach().cpu().numpy(), None
    
    
    def discriminate(self, X):
        X = Tensor(X)
        output = self.discriminator(X)
        return output.detach().numpy()
    
    
    def batch_log_prob(self, z):
        n = len(z) // 2
        cond = Tensor(np.concatenate((np.zeros((n, 1)), np.ones((n, 1)))))
        return  self.gen_prior.log_prob(z) + self.discriminator(self.generator(z, cond)).squeeze()
    

    def log_prob(self, z: torch.FloatTensor, batch_size: Optional[int] = None):
        z_flat = z.reshape(-1, self.latent_dim)
        batch_size = batch_size or self.batch_size
        return torch.cat(list(map(self.batch_log_prob, z_flat.split(batch_size, 0))), 0).reshape(z.shape[:-1])


##################################################
###                                            ###
###                   VAE                      ###
###                                            ###
##################################################


class Encoder(nn.Module):
    def __init__(self, n_inputs, latent_dim, hidden_size=64, batch_norm=True):
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(n_inputs, hidden_size),
                                   nn.Tanh())
        if batch_norm:
            self.model.append(nn.BatchNorm1d(hidden_size, 0.8))
            
        self.fc_mean = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        x = self.model(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_outputs, hidden_size=64, batch_norm=True):
        super(Decoder, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(latent_dim, hidden_size),
                                   nn.Tanh())
        if batch_norm:
            self.model.append(nn.BatchNorm1d(hidden_size, 0.8))
            
        self.model.append(nn.Linear(hidden_size, n_outputs))

    def forward(self, x):
        x = self.model(x)
        return x

    
class VAE(nn.Module):
    def __init__(self, n_input=1, n_output=1, latent_dim=16, hidden_size=64, batch_norm=True):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(n_input, latent_dim, hidden_size, batch_norm)
        self.decoder = Decoder(latent_dim, n_output, hidden_size, batch_norm)
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()
        
        eps = Variable(torch.randn(len(x), self.latent_dim))
        noise = z_mean + std * eps
        
        x_tilda = self.decoder(noise)
        
        return z_mean, z_logvar, x_tilda
    

class CondVAE(nn.Module):
    def __init__(self, n_input=1, n_output=1, latent_dim=16, hidden_size=64, batch_norm=True, n_components=2):
        super(CondVAE, self).__init__()
        
        self.encoder = Encoder(n_input + 1, latent_dim, hidden_size, batch_norm)
        self.decoder = Decoder(latent_dim + 1, n_output, hidden_size, batch_norm)
        
        self.latent_dim = latent_dim
        
    def forward(self, x, cond):
        z_mean, z_logvar = self.encoder(torch.cat((x, cond), dim=1))
        std = z_logvar.mul(0.5).exp_()
        
        eps = Variable(torch.normal(mean=torch.Tensor([(self.latent_dim//2)*[-3] + (self.latent_dim//2)*[3]]*len(x))))#torch.randn(len(x), self.latent_dim))
        noise = z_mean + std * eps
        
        x_tilda = self.decoder(torch.cat((noise, cond), dim=1))
        
        return z_mean, z_logvar, x_tilda


##################################################
###                                            ###
###                   CPDVAE                   ###
###                                            ###
##################################################


class CPDVAE(object):
    def __init__(self, in_dim=1, out_dim=1, batch_size=16, n_epochs=20, prior_weight=0.5, latent_dim=16, learning_rate=1e-3, hidden_size=64, batch_norm=False):
        
        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.latent_dim   = latent_dim
        self.prior_weight = prior_weight
        self.name = 'VAE'
        
        self.vae      = VAE(in_dim, out_dim, latent_dim, hidden_size, batch_norm)
        self.opt_vae  = torch.optim.Adam(self.vae.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        self.loss_history_vae  = []
        
        if cuda:
            self.vae.cuda()
    
    
    def fit(self, X_ref):
        
        real_ref = Tensor(X_ref)
        
        # Turn on training
        self.vae.train(True)
        
        # Fit GAN
        for e in range (self.n_epochs):
            for ref_batch in DataLoader(real_ref, batch_size=self.batch_size, shuffle=True):
                
                # variable_batch = Variable(ref_batch + torch.randn(ref_batch.shape)/100)
                
                mean, logvar, rec_enc = self.vae(ref_batch)#variable_batch)
                
                # Reconstruction loss
                
                Rec_loss = ((rec_enc - ref_batch)**2).mean()
                
                # Prior loss
                
                prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
                prior_loss = torch.mean(-0.5 * torch.sum(prior_loss, dim=1), dim=0)
                
                vae_loss   = self.prior_weight * prior_loss + Rec_loss
                self.loss_history_vae.append(vae_loss.detach().numpy())
                
                self.opt_vae.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.opt_vae.step()
                
        # Turn off training
        self.vae.train(False)
        # plt.plot(self.loss_history_vae)
        # plt.show()
        
        
    def sample(self, n):
        noise = Variable(Tensor(np.random.normal(0, 1, (n, self.latent_dim))))
        X_gen = self.vae.decoder(noise)
        return X_gen.detach().cpu().numpy(), None


class CPDCondVAE(object):
    def __init__(self, batch_size=16, n_epochs=20, in_dim=1, out_dim=1, prior_weight=0.5, latent_dim=16, learning_rate=3e-4, hidden_size=64, batch_norm=True, n_components=2):
        
        self.batch_size   = batch_size
        self.n_epochs     = n_epochs
        self.latent_dim   = latent_dim
        self.prior_weight = prior_weight
        
        self.vae      = CondVAE(in_dim, out_dim, latent_dim, hidden_size, batch_norm, n_components)
        self.opt_vae  = torch.optim.Adam(self.vae.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        self.loss_history_vae  = []
        
        if cuda:
            self.vae.to(device)
    
    
    def fit(self, X, X_cond):
        
        X = Tensor(X)
        X_cond = Tensor(X_cond)
        X_train = TensorDataset(X, X_cond)
        
        # Turn on training
        self.vae.train(True)
        
        # Fit GAN
        for e in range (self.n_epochs):
            for X_batch, cond_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True):
                
                # variable_batch = Variable(ref_batch + torch.randn(ref_batch.shape)/100)
                # variable_batch.to(device)
                
                mean, logvar, rec_enc = self.vae(X_batch, cond_batch)
                
                # Reconstruction loss
                
                Rec_loss = ((rec_enc - X_batch)**2).mean()
                
                # Prior loss
                
                prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
                prior_loss = torch.mean(-0.5 * torch.sum(prior_loss, dim=1), dim=0)
                
                vae_loss   = self.prior_weight * prior_loss + Rec_loss
                # self.loss_history_vae.append(vae_loss.detach().numpy())
                
                self.opt_vae.zero_grad()
                vae_loss.backward()
                self.opt_vae.step()
                
        # Turn off training
        self.vae.train(False)
        # plt.plot(self.loss_history_vae)
        # plt.show()
        
        
    def sample(self, n):
        noise = Variable(Tensor(np.random.normal(0, 1, (2*n, self.latent_dim))))
        cond  = np.concatenate((np.zeros(n), np.ones(n)))
        X     = Variable(Tensor(np.c_[noise, cond]))
        X_gen = self.vae.decoder(X)
        return X_gen.detach().cpu().numpy(), None


##################################################
###                                            ###
###                   NF                       ###
###                                            ###
##################################################


class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super(InvertibleLayer, self).__init__()

        self.var_size = var_size

    def f(self, x):
        pass

    def g(self, x):

        pass


class NormalizingFlow(nn.Module):

    def __init__(self, layers, prior, prior_weight=0.5):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior
        self.prior_weight = prior_weight

    def log_prob(self, x):
        
        log_likelihood = None

        for layer in self.layers:
            x, change = layer.f(x)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self. prior_weight * self.prior.log_prob(x)

        return log_likelihood.mean()

    def sample(self, n):
        
        x = self.prior.sample((n,))
        for layer in self.layers[::-1]:
            x = layer.g(x)

        return x


class RealNVP(InvertibleLayer):

    def __init__(self, var_size, mask, hidden=10):
        super(RealNVP, self).__init__(var_size=var_size)

        self.mask = mask.to(device)

        self.nn_t = nn.Sequential(
            nn.Linear(var_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size)
        )
        self.nn_s = nn.Sequential(
            nn.Linear(var_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size),
        )

    def f(self, x):
        
        t = self.nn_t(x * self.mask[None, :])
        s = self.nn_s(x * self.mask[None, :])

        new_x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        log_det = (s * (1 - self.mask[None, :])).sum(dim=-1)
        
        return new_x, log_det

    def g(self, x):
        
        t = self.nn_t(x * self.mask[None, :])
        s = self.nn_s(x * self.mask[None, :])

        new_x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        
        return new_x


class CPDNF(object):

    def __init__(self, var_size=2, batch_size=16, n_epochs=20, lr=1e-3, n_layers=1, hidden=16, latent_dim=1, prior_weight=0.5):

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_layers = n_layers
        self.hidden = hidden
        self.var_size = var_size
        self.latent_dim = latent_dim
        self.prior_weight = prior_weight
        self.name = 'NF'

        prior = torch.distributions.MultivariateNormal(torch.zeros(self.var_size + latent_dim, device=device), torch.eye(self.var_size + latent_dim, device=device))

        layers = []
        for i in range(self.n_layers):
            layers.append(RealNVP(var_size=self.var_size + latent_dim, mask=((torch.arange(self.var_size + latent_dim) + i) % 2), hidden=self.hidden))

        self.nf = NormalizingFlow(layers=layers, prior=prior, prior_weight=self.prior_weight).to(device)
        self.opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr)

    def fit(self, X):

        # numpy to tensor
        X_real = Tensor(X)
        
        self.loss_history = []

        # Fit NF
        
        for epoch in range(self.n_epochs):
            for X_batch in DataLoader(X_real, batch_size=self.batch_size, shuffle=True):
                
                noise = Tensor(np.random.normal(0, 1, (len(X_batch), self.latent_dim)))

                X_batch = torch.cat((X_batch, noise), dim=1)
                
                # calculate loss
                loss = -self.nf.log_prob(X_batch)

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())

    def sample(self, n):
        
        X_gen = self.nf.sample(n).cpu().detach().numpy()
        
        return X_gen[:, :self.var_size], None
