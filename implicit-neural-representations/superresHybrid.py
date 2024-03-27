#!/home/gundogdu/inr3/bin/python
####################################################################################################
#Implicit Neural Representations for Diffusion MRI  
#Author: Batuhan Gundogdu
####################################################################################################
import warnings
from PIA import hybrid_fit
warnings.filterwarnings("ignore", category=UserWarning)
import scipy.io as sio
import mat73
import numpy as np
import copy
from SRDWI import  ImageFitting_set, Siren, get_mgrid, input_mapping, calculate_ADC, resize_array
import torch
from tqdm import tqdm
from skimage.transform import rescale
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import morphology



def main():

    patient_list = (65, 82, 83, 89, 99, 104)
    headers_for_quantitative_results = 'Pt_id, b-value, te-value, slice, SSIM-spline, SSIM-SR\n'

    pt_id = 99
    #for pt_id in patient_list:
    print(pt_id)

    data_address = os.path.join(f'/home/gundogdu/Desktop/pat{pt_id:03d}', 'master.mat')
    output_address = f'/home/gundogdu/Desktop/SR_Hybrid_results/pat{pt_id:03d}'
    if not os.path.exists(output_address):
        os.makedirs(output_address)

    file = open(os.path.join(output_address, 'ssim_scores.csv'), mode='w')
    file.write(headers_for_quantitative_results)

    print('Loading data')
    try:
        data = sio.loadmat(data_address)
    except NotImplementedError:
        data = mat73.loadmat(data_address) 
    print('Data loaded')
    TE_values = data['TE']
    bvalues = data['b']
    hybrid_raw_all_acq = data['hybrid_raw']

    hybrid_raw = [[None for j in range(4)] for i in range(4)]
    for b in range(4):
        for te in range(4):
            hybrid_raw[b][te] = np.mean(hybrid_raw_all_acq[b][te], -1) if b else hybrid_raw_all_acq[b][te]

    image_shape = hybrid_raw[0][0].shape
    hybrid_raw_norm = copy.deepcopy(hybrid_raw)       
    maxes = np.zeros((4,4))

    
    for b in range(4):
        for te in range(4):
            maxes[b, te] = hybrid_raw_norm[b][te].max()
            hybrid_raw_norm[b][te] = hybrid_raw_norm[b][te]/maxes[b, te]

    number_of_epochs = 2500
    hidden_dim = 512
    num_layers = 3
    mapping_size = 128
    scale = 0.5
    roi_start_x = 35
    roi_end_x = 95
    roi_start_y = 35
    roi_end_y = 95
    size_x = roi_end_x - roi_start_x
    size_y = roi_end_y - roi_start_y
    recon_hybrid = np.zeros((size_x*2, size_y*2, hybrid_raw_norm[0][0].shape[2], 4, 4))
    for te_value in range(4):

        mean_img = np.zeros((image_shape[0], image_shape[1], image_shape[2], 4))

        for b in range(4):
            mean_img[:, :, :, b] = hybrid_raw_norm[b][te_value]
    

        dataset = ImageFitting_set([mean_img[roi_start_x:roi_end_x:2, roi_start_y:roi_end_y:2, :]])
        dimension = len(dataset.shape)
        B_gauss = np.random.normal(size=(mapping_size, dimension))
        B = torch.from_numpy(B_gauss * scale).float().cuda()

        INR = Siren(in_features=2*mapping_size, out_features=1, 
                        hidden_features=hidden_dim,
                        hidden_layers=num_layers)

        
        INR.cuda()
        inr_params = list(INR.parameters())
        inr_optim = torch.optim.Adam(lr=1e-4, params=inr_params)       
        LR_ground_truth, model_input  = dataset.pixels[0].cuda(), dataset.coords[0].cuda()
        model_input = input_mapping(model_input, B)

        test_input_shape = (size_x*2, size_y*2, mean_img.shape[2], mean_img.shape[3])
        test_input  = input_mapping(get_mgrid(test_input_shape).cuda(), B)


        print(f'Training the INR and Perturb-Net for te = {te_value}')

        for _ in tqdm(range(number_of_epochs)):
            model_output = INR.forward(model_input)
            loss = ((model_output - LR_ground_truth)**2).mean()
            inr_optim.zero_grad()
            loss.backward()
            inr_optim.step()
        print('Training finished, now creating outputs')

        del model_input, model_output, LR_ground_truth
        torch.cuda.empty_cache()
        recon = torch.clamp(INR.forward(test_input), min=0).cpu().view(test_input_shape).detach().numpy()

        for b in range(4): 
            recon_hybrid[:, : , :, b, te_value] = (recon[:, :, :, b])*maxes[b, te_value] 

        del test_input
        torch.cuda.empty_cache()


    _slice = 9
    hybrid_normalized = np.zeros_like(recon_hybrid)

    for b in range(4):
        for te in range(4):
            hybrid_normalized[:, :, :, b, te] = 1000*recon_hybrid[:, :, :, b, te]/(recon_hybrid[:, :, :, 0, 0] + 1e-7)
    hybrid_data2 = np.reshape(hybrid_normalized, (size_x*2, size_y*2, hybrid_normalized.shape[2], 16))
    model_input = np.squeeze(hybrid_data2[: , : , _slice, :])
    bins = (model_input.shape[0], model_input.shape[1])
    model_input = np.reshape(model_input, (model_input.shape[0]*model_input.shape[1], 16))
    

    D, T2, v = hybrid_fit(model_input)

    fig, ax = plt.subplots(3,3, figsize=(15,15))
    for r in range(3):
        for c in range(3):
            if r==0:
                x_image = v
                title = ['epithelium volume', 'stroma volume', 'lumen volume']
                ylims = [(0,1), (0,1), (0,1)]
            elif r==1:
                x_image = D
                title = ['epithelium ADC', 'stroma ADC', 'lumen ADC']
                ylims = [(0.3, 0.7), (0.7, 1.7), (2.7, 3)]

            else:
                x_image =  T2
                title = ['epithelium T2', 'stroma T2', 'lumen T2']
                ylims = [(20, 70), (40, 100), (500, 1000)]

            im = ax[r,c].imshow(np.reshape(x_image[:,c], bins), vmin=ylims[c][0], vmax=ylims[c][1],cmap='jet')
            ax[r,c].set_title(fr'{title[c]}')
            divider = make_axes_locatable(ax[r,c])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax)
    v_ep = np.reshape(v[:,0], bins) 
    v_lu = np.reshape(v[:,2], bins)
    cancer = (v_ep > 0.4)*(v_lu <= 0.2)

    filename = os.path.join(output_address, f"Hybrid{_slice}_m_{mapping_size}_s_{scale}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    adc_map = calculate_ADC(bvalues, np.squeeze(recon_hybrid[:,  : , _slice, :, 0]))
    fig, ax = plt.subplots(1, figsize=(6,6))
    fig.suptitle('predicted cancer map')
    ax.imshow(adc_map, vmax=3, vmin=0, cmap='gray')
    cancer_map = cancer.astype(float)
    cancer_map = morphology.remove_small_objects(cancer_map.astype(bool), min_size=12, connectivity=1)
    cancer_map = cancer_map.astype(float)
    cancer_map[cancer_map==0] = np.nan
    ax.imshow(cancer_map, cmap='autumn',alpha = 0.4)
    ax.axis('off')


    filename = os.path.join(output_address, f"cancer{_slice}_m_{mapping_size}_s_{scale}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
    plt.close()


        
if __name__ == "__main__":
    main()

