import os
#import warnings
import numpy as np
import scipy.io as sio
import mat73
#import torch
import copy
#from INRmodel import calculate_combinations, ImageFitting_set, Siren, PN, input_mapping, get_mgrid, calculate_ADC, resize_array
from tqdm import tqdm
import matplotlib.pyplot as plt
#from skimage.transform import rescale
#from skimage.metrics import structural_similarity as ssim

#warnings.filterwarnings("ignore", category=UserWarning)

def main():

    roi_start = 40
    roi_end = 90 
    patient_list = (65, 82, 83, 89, 99, 104)
    for pt_id in patient_list:
        print(pt_id)
        data_address = f'/home/gundogdu/Desktop/pat{pt_id:03d}/master2.mat'
        output_address = f'/home/gundogdu/Desktop/SR_results_testLR/pat{pt_id:03d}'
        if not os.path.exists(output_address):
            os.makedirs(output_address)

        print('Loading data')
        try:
            data = sio.loadmat(data_address)
        except NotImplementedError:
            data = mat73.loadmat(data_address)
        print('Data loaded')
        hybrid_raw_orig = data['hybrid_raw']
        image_shape = hybrid_raw_orig[0][0].shape

        hybrid_orig_norm = copy.deepcopy(hybrid_raw_orig)
        maxes = np.zeros((4,4))

        for b in range(4):
            for te in range(4):
                maxes[b, te] = hybrid_orig_norm[b][te].max()
                hybrid_orig_norm[b][te] = hybrid_orig_norm[b][te]/maxes[b, te]

        mean_img2 = np.zeros((image_shape[0], image_shape[1], image_shape[2], 4))
        for b in range(4):
            mean_img2[:, :, :, b] = hybrid_orig_norm[b, 0] if not b else np.squeeze(np.mean(hybrid_orig_norm[b, 0], -1))


        bvalues = data['b']    
        for _slice in tqdm(range(4, mean_img2.shape[2])):
            for b in range(4): 
                _, ax = plt.subplots(1,3, figsize=(30,10))
                ax[0].imshow(mean_img2[roi_start:roi_end:2,roi_start:roi_end:2, _slice, b], cmap='gray')
                ax[0].set_title(f'LR b={bvalues[b]} $s/mm^2$')
                ax[1].imshow(mean_img2[roi_start:roi_end:2,roi_start:roi_end:2, _slice, b], cmap='gray')
                ax[1].set_title(f'LR b={bvalues[b]} $s/mm^2$')
                ax[2].imshow(mean_img2[roi_start:roi_end:2,roi_start:roi_end:2, _slice, b], cmap='gray')
                ax[2].set_title(f'LR b={bvalues[b]} $s/mm^2$')
                for axi in range(3):
                    ax[axi].axis('off')
                filename = os.path.join(output_address, f"slice_{_slice}_b_{b}.png")
                plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
                plt.close()

    
    print('Done')


if __name__ == "__main__":
    main()   