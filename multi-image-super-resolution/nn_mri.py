import SimpleITK as sitk
import os
import numpy as np
import scipy.io as sio

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
    def __init__(self, pt_id, b, cancer_loc, contralateral_loc, cancer_slice, acquisitions):
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
        self.cancer_slice = cancer_slice
        self.acquisitions = acquisitions
        self.b = b
        pt_no = self.pt_id.split('-')[-1]
        eps = 1e-7 #to avoid division by zero and log of zero errors in ADC calculation
        filename = '../anon_data/pat' + pt_no + '_alldata.mat'
        self.dwi = sio.loadmat(filename)['data']
        filename = '../anon_data/pat' + pt_no + '_mean_b0.mat'
        self.b0 = sio.loadmat(filename)['data_mean_b0']
        filename = '../anon_data/pat' + pt_no + '_ADC_alldata_mm.mat'
        self.accept = np.ones(self.dwi.shape, dtype=int)
        try :
            self.adc = sio.loadmat(filename)['ADC_alldata_mm']
        except OSError:
            rep_b0 = np.transpose(np.tile(self.b0,(self.dwi.shape[-1],1,1,1)),(1,2,3,0))
            self.adc = -(np.log(self.dwi/(rep_b0 + 1e-7) + 1e-7)/self.b)*1000

                         
cases = []

#cases.append(case('18-1681-07', 900, (67, 71), (67, 59), 11, (8, 8, 8)))
cases.append(case('18-1681-08', 900, (79, 71), (79, 57), 10, (8, 7, 8)))
#cases.append(case('18-1681-09', 900, (63, 63), (63, 55), 15, (8, 8, 8)))
cases.append(case('18-1681-30', 900, (66, 56), (66, 73), 17, (8, 8, 8)))
#cases.append(case('18-1681-37', 900, (70, 70), (70, 61), 10, (8, 8, 8)))
#cases.append(case('17-1694-55', 1500, (60, 57), (60, 69), 13, (4, 4, 4)))
cases.append(case('18-1681-41', 1500, (69, 57), (69, 69), 8, (4, 4, 4)))
#cases.append(case('18-1681-40', 1500, (72, 72), (72, 56), 4, (4, 4, 4)))
#cases.append(case('18-1681-45', 1500, (71, 68), (70, 58), 12, (4, 4, 4)))
cases.append(case('18-1681-47', 1500, (74, 48), (74, 82), 10, (4, 4, 4)))
