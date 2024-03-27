####################################################################################################
#Implicit Neural Representations for MRI  
#Author: Batuhan Gundogdu
####################################################################################################
import scipy.io as sio
import torch
from torch import nn
import numpy as np
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from PIL import Image#, ImageSequence
import matplotlib#; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import SimpleITK as sitk

def save_dicom(img, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    expanded = np.expand_dims(img, 0).astype('int16')
    filtered_image = sitk.GetImageFromArray(expanded)
    writer = sitk.ImageFileWriter()
    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:,:,i]
        writer.SetFileName(filename)
        writer.Execute(image_slice)

class case:
    def __init__(self, pt_id, b, cancer_loc, contralateral_loc, noise, cancer_slice, acquisitions):
        '''
        class for a case
        pt_id : the patient id
        cancer_loc : cancer location pixel (center of a 3x3 region)
        contralateral_loc : mirror of the cancer location with is non-cancer
        cancer_slice : slice number where the cancer exists
        acquisitions : number of slices per X, Y, Z directions 
        '''
        self.pt_id = pt_id
        self.cancer_loc = cancer_loc
        self.contralateral_loc = contralateral_loc
        self.noise = noise
        self.cancer_slice = cancer_slice
        self.acquisitions = acquisitions
        self.b = b
        pt_no = self.pt_id.split('-')[-1]
        eps = 1e-7 #to avoid division by zero and log of zero errors in ADC calculation
        filename = '../anon_data/pat' + pt_no + '_alldata.mat'
        self.dwi = sio.loadmat(filename)['data']
        filename = '../anon_data/pat' + pt_no + '_mean_b0.mat'
        self.b0 = sio.loadmat(filename)['data_mean_b0']
        self.accept = np.ones(self.dwi.shape, dtype=int)
        filename = '../anon_data/pat' + pt_no + '_ERD.mat'
        self.erd = sio.loadmat(filename)['ADC_alldata_mm_ERD']




def calculate_contrast(case, scale, image, focus):
    """ calculates the contrast between the cancer and the collateral benign tissue"""
    # cancer center x,y locations
    cc_x, cc_y = tuple((i -focus)*scale for i in case.cancer_loc)
    # contralateral benign x,y locations
    cb_x, cb_y = tuple((i -focus)*scale for i in case.contralateral_loc)
    # noise x,y locations
    cn_x, cn_y = tuple((i -focus)*scale for i in case.noise)
    cancer_area = image[cc_x - scale : cc_x + scale, cc_y - scale : cc_y + scale]
    contralateral_area = image[cb_x - scale : cb_x + scale, cb_y - scale : cb_y + scale]
    noise_area = image[cn_x - scale : cn_x + scale, cn_y - scale : cn_y + scale]
    
    #cancer_mean
    cm = cancer_area.mean()
    #begign_mean
    bm = contralateral_area.mean()
    #cancer_variance
    varc = np.std(cancer_area)**2
    #benign variance
    varb = np.std(contralateral_area)**2
    #noise variance
    varn = np.std(noise_area)
    
    C = (cancer_area.mean() / (contralateral_area.mean() + 1e-7))
    CNR = abs(cancer_area.mean() - contralateral_area.mean())/np.sqrt(varc + varb)
    CNR2 = abs(cancer_area.mean() - contralateral_area.mean())/varn
    return C, CNR, CNR2

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)  
    return mgrid

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
      

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                               np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
          return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features, 
                 first_omega_0=30., 
                 hidden_omega_0=30.):
        super().__init__()
        # self.net is the INR that calculates signal intensities for its inputs
        self.net = []
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        self.net.append(self.final_linear)

        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(False)
        output = self.net(coords)

        return output

class PN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.tanh = nn.Tanh()  
        self.perturb_linear = nn.Linear(in_features + 1, hidden_features)
        self.perturb_linear2 = nn.Linear(hidden_features, 2)
        
    def forward(self, coords, sample=0, eps=0):
        coords = coords.clone().detach().requires_grad_(False)
        acq = torch.tensor([sample/10.], dtype=torch.float).cuda()
        acq = acq.repeat(coords.size(0),1)
        perturbation = self.perturb_linear(torch.cat((coords, acq),-1))
        perturbation = self.tanh(perturbation)
        perturbation = self.perturb_linear2(perturbation)
        pertubation = eps*self.tanh(perturbation)
        
        return pertubation
    
# Fourier feature mapping
def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = torch.matmul(2.*np.pi*x, B.T)
    return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

def get_image_tensor(img):
    
    sidelength, _ = img.size
    #transform = Compose([Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    transform = Compose([Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    img = transform(img)
    return img

class ImageFitting_set(Dataset):
    
    "This is rearranged for MR dataset"
 
    def __init__(self, img_dataset):
        super().__init__()
        sidelength, sidelength = img_dataset[0].size
        self.orig = np.empty((len(img_dataset),img_dataset[0].size[0],img_dataset[0].size[1]))
        self.pixels = torch.empty((len(img_dataset),sidelength**2, 1))
        self.coords = torch.empty((len(img_dataset),sidelength**2, 2))
        for ctr, img in enumerate(img_dataset):
            self.orig [ctr] = np.array(img)
            img = get_image_tensor(img)
            self.pixels[ctr] = img.permute(1, 2, 0).view(-1, 1)
            self.coords[ctr] = get_mgrid(sidelength, 2)
            self.mean = sum(self.orig)/len(self.orig)
            self.shape = img_dataset[0].size
    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return self.coords, self.pixels

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



def save_fig(array,filename,size=(2,2),dpi=600, cmap='gray'):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap(cmap)
    ax.imshow(array, aspect='equal')
    plt.savefig(filename, format='eps', dpi=dpi)
