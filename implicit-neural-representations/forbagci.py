#!/home/gundogdu/inr3/bin/python

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import scipy.io as sio
import mat73
import numpy as np
import copy
from multiprocessing import Pool
from SRDWI import calculate_combinations, ImageFitting_set, Siren, PN, input_mapping
import torch
from tqdm import tqdm
from skimage.transform import rescale
from skimage.metrics import structural_similarity as ssim
import os




def main():

    patient_list = (65, 82, 83, 89, 99, 104)
    headers_for_quantitative_results = 'Pt_id, b-value, slice, SSIM-spline, SSIM-SR\n'
    file = open(f'/home/gundogdu/Desktop/SR_results_for_bagci/ssim_scores.csv', mode='w')
    file.write(headers_for_quantitative_results)
    gt_dataset = []
    lr_dataset = []
    zero_shot_SR = []

    for pt_id in patient_list:
        print(pt_id)
        data_address = f'/home/gundogdu/Desktop/pat{pt_id:03d}/master.mat'
        output_address = f'/home/gundogdu/Desktop/SR_results_for_bagci/pat{pt_id:03d}'

        if not os.path.exists(output_address):
            os.makedirs(output_address)

        print('Loading data')
        try:
            data = sio.loadmat(data_address)
        except NotImplementedError:
            data = mat73.loadmat(data_address) 
        print('Data loaded')
        hybrid_raw = data['hybrid_raw']
        image_shape = hybrid_raw[0][0].shape
        
        hybrid_raw_norm = copy.deepcopy(hybrid_raw)

        maxes = np.zeros((4,4))

        for b in range(4):
            for te in range(4):
                maxes[b, te] = hybrid_raw_norm[b][te].max()
                hybrid_raw_norm[b][te] = hybrid_raw_norm[b][te]/maxes[b, te]

        num_processes = 32
        pool = Pool(processes=num_processes)
        
        voxel_list = [(i, j, k) for i in range(image_shape[0])
                                for j in range(image_shape[1])
                                for k in range(image_shape[2])]
        print('Calculating all combinations in parallel')
        results = pool.starmap(calculate_combinations, [(voxel, hybrid_raw_norm) for voxel in voxel_list])
        pool.close()
        pool.join()

        acquisitions = np.zeros((image_shape[0], 
                    image_shape[1], 
                    image_shape[2],
                    4,
                    hybrid_raw[1][0].shape[3]*hybrid_raw[2][0].shape[3]*hybrid_raw[3][0].shape[3]))

        for idx, voxel in enumerate(voxel_list):
            i, j, k = voxel
            acquisitions[i, j, k, :, :] = results[idx]

        print('Done calculating')

        select_acquisitions = np.squeeze(acquisitions)

        mean_img = np.mean(select_acquisitions,-1)

        number_of_epochs = 2500
        pertubation_epochs = 10 #This corresponds to 1000 steps total for each acquisition
        hidden_dim = 512
        num_layers = 3
        PN_dim = 128
        roi_start = 40
        roi_end = 90

        img_dataset = []
        for inx in range(select_acquisitions.shape[-1]):
            img = select_acquisitions[roi_start:roi_end:2, roi_start:roi_end:2, :, :, inx]
            img_dataset.append(img)

        mean_dataset = ImageFitting_set([mean_img[roi_start:roi_end:2, roi_start:roi_end:2, :]])
        dataset = ImageFitting_set(img_dataset)
        dimension = len(dataset.shape)
        HR = ImageFitting_set([mean_img[roi_start:roi_end, roi_start:roi_end, :]])

        mapping_size = 128
        scale = 0.5

        B_gauss = np.random.normal(size=(mapping_size, dimension))
        B = torch.from_numpy(B_gauss * scale).float().cuda()

        INR = Siren(in_features=2*mapping_size, out_features=1, 
                        hidden_features=hidden_dim,
                        hidden_layers=num_layers)
        INR.cuda()
        PerturbNet = PN(in_features=2*mapping_size, hidden_features=PN_dim, dimension=dimension)
        PerturbNet.cuda()

        inr_params = list(INR.parameters())
        inr_optim = torch.optim.Adam(lr=1e-4, params=inr_params)
        perturb_params = list(PerturbNet.parameters())
        perturb_optim = torch.optim.Adam(lr=1e-6, params=perturb_params)
        torch.cuda.empty_cache()

        LR_ground_truth, model_input  = mean_dataset.pixels[0].cuda(), mean_dataset.coords[0].cuda()
        model_input = input_mapping(model_input, B)
        HR_model_input  = HR.coords[0].cuda()
        HR_model_input = input_mapping(HR_model_input, B)

        HR_img = mean_img[roi_start:roi_end, roi_start:roi_end, :, :]

        print(f'Training the INR and Perturb-Net for mapping size = {mapping_size} and sigma = {scale}')

        for ctr in tqdm(range(number_of_epochs)):
            if ctr < number_of_epochs - pertubation_epochs:
                model_output = INR.forward(model_input)
                loss = ((model_output - LR_ground_truth)**2).mean()
                inr_optim.zero_grad()
                loss.backward()
                inr_optim.step()
            else:
                if ctr%2:
                    model_output = INR.forward(model_input)
                    loss = ((model_output - LR_ground_truth)**2).mean()
                    inr_optim.zero_grad()
                    loss.backward()
                    inr_optim.step()
                else:
                    for sample in range(len(dataset)):
                        ground_truth  = dataset.pixels[sample].cuda()
                        perturbed_input = PerturbNet.forward(model_input, sample, 1/128.)
                        perturbed_input = input_mapping(perturbed_input, B)
                        model_output = INR.forward(perturbed_input)

                        loss = ((model_output - ground_truth)**2).mean()
                        perturb_optim.zero_grad()
                        loss.backward()
                        perturb_optim.step()

        print('Training finished, now creating outputs')
        
        bvalues = data['b']
        SR_recon = torch.clamp(INR.forward(HR_model_input), min=0).cpu().view(HR.shape).detach().numpy()
        
        for _slice in tqdm(range(10, 21)):
            b = 3 
            HR_ref = HR_img[:, :, _slice, b]
            HR_ref = HR_ref/HR_ref.max()
            LR_ref = HR_ref[::2, ::2]
            spline = rescale(LR_ref, 2, anti_aliasing=True)
            spline = spline/spline.max()
            SR = SR_recon[:, :, _slice, b]
            SR = SR/SR.max()
            mask = HR_ref > 0.05 # As suggested by Gourdeu et. al (2022) Medical Physics
            row_to_file = f'{pt_id}, {bvalues[b]}, {_slice}, {ssim(HR_ref*mask, spline*mask, data_range=1)}, {ssim(HR_ref*mask, SR*mask, data_range=1)}\n'
            file.write(row_to_file)
            gt_dataset.append(HR_ref)
            lr_dataset.append(LR_ref)
            zero_shot_SR.append(SR)

    np.savez(f'/home/gundogdu/Desktop/SR_results_for_bagci/zero_shot_dwi.npz', lr_dataset=lr_dataset, 
             gt_dataset=gt_dataset, zero_shot_SR=zero_shot_SR)
    print('Done')
    


if __name__ == "__main__":
    main()
