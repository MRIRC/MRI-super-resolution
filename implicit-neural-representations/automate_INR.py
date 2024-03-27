####################################################################################################
#Implicit Neural Representations for Diffusion MRI  
#Author: Batuhan Gundogdu
####################################################################################################

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import scipy.io as sio
from nn_mri import Siren, PN, input_mapping
from SRDWI import get_mgrid, ImageFitting_set
import numpy as np
import torch
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"]="3"


filename = '/home/gundogdu/toy2.mat'
acquisitions = sio.loadmat(filename)['pertubed_acq']

mapping_size = 128
scale = 2.0

B_gauss = np.random.normal(size=(mapping_size, 2))
B = torch.from_numpy(B_gauss * scale).float().cuda()



mean_img = np.mean(acquisitions,-1)

img_dataset = []
for inx in range(acquisitions.shape[-1]):
    img = acquisitions[:,:,inx]
    img_dataset.append(img)

mean_dataset = ImageFitting_set([mean_img])
dataset = ImageFitting_set(img_dataset)

INR = Siren(in_features=2*mapping_size, out_features=1, 
                    hidden_features=128,
                    hidden_layers=3)
INR.cuda()
PerturbNet = PN(in_features=2*mapping_size, hidden_features=128)
PerturbNet.cuda()

inr_params = list(INR.parameters())
inr_optim = torch.optim.Adam(lr=1e-4, params=inr_params)
perturb_params = list(PerturbNet.parameters())
perturb_optim = torch.optim.Adam(lr=1e-6, params=perturb_params)
torch.cuda.empty_cache()

new_loss = 1000
sr_epochs = np.zeros((256, 256, 50))
img_ctr = 0
for ctr in tqdm(range(2000)):

    if ctr < 500:
        # TODO: It is slowing me down to try to read these to gpu memory every time, gonna fix it
        ground_truth, model_input  = mean_dataset.pixels[0], mean_dataset.coords[0]
        ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
        model_input = input_mapping(model_input, B)
        ground_truth /= ground_truth.max()
        model_output = INR.forward(model_input)
        loss = ((model_output - ground_truth)**2).mean()
        inr_optim.zero_grad()
        loss.backward()
        inr_optim.step()

    else:
        if ctr%2:
            ground_truth, model_input  = mean_dataset.pixels[0], mean_dataset.coords[0]
            ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
            model_input = input_mapping(model_input, B)
            ground_truth /= ground_truth.max()
            model_output = INR.forward(model_input)
            loss = ((model_output - ground_truth)**2).mean()
            inr_optim.zero_grad()
            loss.backward()
            inr_optim.step()
        # else:
        #     for sample in range(len(dataset)):
        #         ground_truth, model_input  = dataset.pixels[sample], dataset.coords[sample]
        #         ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
        #         model_input = input_mapping(model_input, B)
        #         ground_truth /= ground_truth.max()
        #         perturbed_input = PerturbNet.forward(model_input, sample, 1/128.)
        #         perturbed_input = input_mapping(perturbed_input, B)
        #         model_output = INR.forward(perturbed_input)
        #         if not sample:
        #             loss = ((model_output - ground_truth)**2).mean()
        #         else:
        #             loss += ((model_output - ground_truth)**2).mean()
        
        #     perturb_optim.zero_grad()
        #     loss.backward()
        #     perturb_optim.step()
    if not ctr%100:
        # TODO: observe training and validation loss somewhere here, based on the ground truth cameraman
        model_input  = get_mgrid((256, 256)).cuda()
        model_input  = input_mapping(model_input, B)
        recon = INR.forward(model_input).cpu().view(256,256).detach().numpy()
        sr_epochs[:, :, img_ctr] = recon
        img_ctr += 1

    # if loss.item() < new_loss:
    #     new_loss = loss.item()

print(f'Done {scale} and {mapping_size}')
sio.savemat(f'nonPILoutput_b_{scale}_emb_{mapping_size}.mat', {'recon':recon})

#PATH = 'toy_model.pt'
#torch.save(INR.state_dict(), PATH)