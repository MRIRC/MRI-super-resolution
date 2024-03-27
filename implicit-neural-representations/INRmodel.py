####################################################################################################
#Implicit Neural Representations for Diffusion MRI  
#Author: Batuhan Gundogdu
####################################################################################################
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
from scipy.interpolate import interp1d
import itertools

def get_mgrid(shape):
    ''' returns mash grid input values for a tensor of given shape'''
    dim = len(shape)
    tensors = tuple(tuple([torch.linspace(-1, 1, steps=len) for len in shape]))
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)  
    return mgrid

class ImageFitting_set(Dataset):
    
    "Rearranged for 3D MRI dataset"
 
    def __init__(self, img_dataset):
        super().__init__()
        shape = img_dataset[0].shape
        self.shape = shape
        self.pixels = torch.empty((len(img_dataset), np.prod(shape), 1))
        self.coords = torch.empty((len(img_dataset), np.prod(shape), len(shape)))
        for ctr, img in enumerate(img_dataset):
            img = torch.from_numpy(img).float()
            self.pixels[ctr] = img.contiguous().view(-1).unsqueeze(dim=-1)
            self.coords[ctr] = get_mgrid(shape)
        
    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return self.coords, self.pixels

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
    
class ComplexGaborLayer2D(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first     
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features,
                                    out_features,
                                    bias=bias,
                                    dtype=dtype)
        
        def init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features,1 / self.in_features)
                    self.scale_orth.weight.uniform_(-1 / self.in_features,1 / self.in_features)
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega0,
                                                np.sqrt(6 / self.in_features) / omega0)
                    self.scale_orth .weight.uniform_(-np.sqrt(6 / self.in_features) / omega0,
                                                np.sqrt(6 / self.in_features) / omega0)

    
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        freq_term = torch.exp(1j*self.omega_0*lin)
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)
                
        return freq_term*gauss_term

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, hidden_omega_0=30.):
        super().__init__()
        # self.net is the INR that calculates signal intensities for its inputs
 
        
        self.net = []
        

        self.net.append(SineLayer(in_features,
                                    hidden_features, 
                                    is_first=True))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False))
        #dtype = torch.cfloat
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(self.final_linear)

        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(False)
        output = self.net(coords)

        return  output

class PN(nn.Module):
    def __init__(self, in_features, hidden_features, dimension):
        super().__init__()
        self.tanh = nn.Tanh()  
        self.perturb_linear = nn.Linear(in_features + 1, hidden_features)
        self.perturb_linear2 = nn.Linear(hidden_features, dimension)
        
    def forward(self, coords, sample=0, eps=0):
        coords = coords.clone().detach().requires_grad_(False)
        acq = torch.tensor([sample/10.], dtype=torch.float).cuda()
        acq = acq.repeat(coords.size(0),1)
        perturbation = self.perturb_linear(torch.cat((coords, acq),-1))
        perturbation = self.tanh(perturbation)
        perturbation = self.perturb_linear2(perturbation)
        pertubation = eps*self.tanh(perturbation)
        
        return pertubation
    
def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = torch.matmul(2.*np.pi*x, B.T)
    return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
  
def calculate_ADC(bvalues, slicedata):
    min_adc = -10
    max_adc = 3.0
    eps = 1e-7
    numrows, numcols, _ = slicedata.shape
    adc_map = np.zeros((numrows, numcols))
    for row in range(numrows):
        for col in range(numcols):
            ydata = np.squeeze(slicedata[row, col, :])
            adc = np.polyfit(bvalues.flatten()/1000, np.log(ydata + eps), 1)
            adc = -adc[0]
            adc_map[row, col] =  max(min(adc, max_adc), min_adc)
    return adc_map

def resize_array(arr, new_size=128, kind='cubic'):
    old_shape = arr.shape
    new_shape = (old_shape[0], old_shape[1], new_size)
    new_arr = np.zeros(new_shape)
    x_old = np.linspace(0, 1, old_shape[2])
    x_new = np.linspace(0, 1, new_size)
    f = interp1d(x_old, arr, kind=kind, axis=2)
    for i in range(new_size):
        new_arr[:, :, i] = f(x_new[i])
    return new_arr

def calculate_combinations(voxel, hybrid_raw_norm):
    i, j, k = voxel
    te = 0 # Note we are choosing TE=70ms becuase it is the one closest to the clinical DWI
    b0 = [hybrid_raw_norm[0][te][i, j, k]]
    b1 = [x for x in hybrid_raw_norm[1][te][i, j, k, :]]
    b2 = [x for x in hybrid_raw_norm[2][te][i, j, k, :]]
    b3 = [x for x in hybrid_raw_norm[3][te][i, j, k, :]]
    all_bs = [b0, b1, b2, b3]
    combs = np.asarray(list(itertools.product(*all_bs))).T
    return combs
