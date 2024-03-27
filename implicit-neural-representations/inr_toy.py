import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from nn_mri import ImageFitting_set, SineLayer, get_mgrid
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb
from torch import nn
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.cluster import AgglomerativeClustering
import scipy.io as sio
import os
import argparse
from csv import writer

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features, 
                 first_omega_0=30., 
                 hidden_omega_0=30.,
                 perturb=False):
        super().__init__()
        # self.net is the INR that calculates signal intensities for its inputs
        self.net = []
        self.tanh = nn.Tanh()  
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        self.net.append(self.final_linear)

        self.net = nn.Sequential(*self.net)
        self.perturb_linear = nn.Linear(in_features + 1, in_features + 1)
        self.perturb_linear2 = nn.Linear(in_features + 1, in_features)
        
        self.perturb = perturb
        
    def forward(self, coords, sample=0,eps=0):
        coords = coords.clone().detach().requires_grad_(False) # allows to take derivative w.r.t. input
        if self.perturb:
            acq = torch.tensor([sample], dtype=torch.float).cuda()
            acq = acq.repeat(coords.size(0),1)
            perturbation = self.perturb_linear(torch.cat((coords, acq),-1))
            perturbation = self.tanh(perturbation)
            perturbation = self.perturb_linear2(perturbation)
            pertubation = eps*self.tanh(perturbation)
            coords = coords + pertubation
        output = self.net(coords)

        return output
    
filename = '/home/gundogdu/toy.mat'
acquisitions = 1 - sio.loadmat(filename)['pertubed_acq']
img_dataset = []
mean_img = np.mean(acquisitions,-1)
dataset = ImageFitting_set([Image.fromarray(mean_img)])

dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
img_siren = Siren(in_features=2, out_features=1, 
                      hidden_features=128,
                      hidden_layers=3, perturb=False)

img_siren.cuda()


params = list(img_siren.net.parameters())
optim = torch.optim.Adam(lr=3e-4, params=params)
torch.cuda.empty_cache()
ctr = 0
new_loss = 1000
while True:
    for sample in range(len(dataset)):
        ground_truth, model_input  = dataset.pixels[sample], dataset.coords[sample]
        ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
        ground_truth /= ground_truth.max()
        model_output = img_siren.forward(model_input, sample, 4/720.)
        if not sample:
            loss = ((model_output - ground_truth)**2).mean()
        else:
            loss += ((model_output - ground_truth)**2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (loss.item() > new_loss and ctr>100) or loss.item() < 1e-9 :
        break      
    else:
        new_loss = loss.item()
    if not ctr%1000:
        print(new_loss)
        model_input  = get_mgrid(720, 2).cuda()
        recon = img_siren.forward(model_input, 0, 4/720.0).cpu().view(720,720).detach().numpy()
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].imshow(recon, vmin=0.6, vmax=1, cmap='gray')
        ax[0].set_title('super')
        ax[1].imshow(rescale(mean_img,4), vmin=0.6, vmax=1,cmap='gray')
        ax[1].set_title('mean')
        # display.display(plt.gcf())
    ctr +=1
print('Done')

PATH = 'toy_model.pt'
torch.save(img_siren.state_dict(), PATH)