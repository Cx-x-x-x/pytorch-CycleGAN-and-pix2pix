import numpy as np
import torch
import scipy
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import scipy.io as scio
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


def np_range_norm(image, maxminnormal=True, range1=True):
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if maxminnormal:
            _min = image.min()
            _range = image.max() - image.min()
            normal_image = (image - _min) / _range
            if range1:
                normal_image = (normal_image - 0.5) * 2
        else:
            _mean = image.mean()
            _std = image.std()
            normal_image = (image - _mean) / _std

    return normal_image


def make_ynet_dataset(dir, max_dataset_size=float("inf")):
    mats = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            mats.append(path)
    return mats[:min(max_dataset_size, len(mats))]


class YnetDataset(BaseDataset):
    __inputdata = []
    __inputimg = []
    __outputdata = []

    def __init__(self, opt, train=True, das=True, transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__inputimg = []

        self.root = opt.dataroot
        self.transform = transform
        self.train = train

        folder = opt.dataroot + '//'

        for file in os.listdir(folder):
            # print(file)
            matdata = scio.loadmat(folder + file)
            self.__inputdata.append(np.transpose(matdata['sensor_data'])[np.newaxis, :, :])
            self.__outputdata.append(matdata['p0'][np.newaxis, :, :])
            self.__inputimg.append(matdata['p0_tr'][np.newaxis, :, :])

        self.paths = sorted(make_ynet_dataset(opt.dataroot, opt.max_dataset_size))
        self.size = len(self.paths)

    def __getitem__(self, index):

        path = self.paths[index % self.size]

        rawdata = self.__inputdata[index]  # .reshape((1,1,2560,120))
        # rawdata = (rawdata-(np.min(np.min(rawdata,axis=2)))/((np.max(np.max(rawdata,axis=2)))-(np.min(np.min(rawdata,axis=2))))
        # rawdata = rawdata -0.5
        # rawdata = np_range_norm(rawdata,maxminnormal=True)
        reconstruction = self.__outputdata[index]  # .reshape((1,1,2560,120))
        # reconstruction = np_range_norm(reconstruction,maxminnormal=True)
        beamform = self.__inputimg[index]

        rawdata = torch.Tensor(rawdata)
        reconstructions = torch.Tensor(reconstruction)
        beamform = torch.Tensor(beamform)

        return {'reimg': reconstruction, 'bfimg': beamform, 'raw': rawdata, 'path': path}

    def __len__(self):
        return len(self.__inputdata)


# if __name__ == "__main__":
#     dataset_pathr = 'D:/model enhanced beamformer/data/20181219/'
#
#     mydataset = ReconDataset(dataset_pathr, train=False, das=True)
#     # print(mydataset.__getitem__(3))
#     train_loader = DataLoader(
#         mydataset,
#         batch_size=1, shuffle=True)
#     batch_idx, (rawdata, reimage, bfim) = list(enumerate(train_loader))[0]
#     print(rawdata.size())
#     print(rawdata.max())
#     print(rawdata.min())
#     print(mydataset.__len__())






