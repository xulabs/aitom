import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
    
import os
import h5py
import numpy as np
from torch.utils.data import Dataset

class CryoET_Patch(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, patch_size=512):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.sample_list = []

        if self.split == 'train':
            list_path = os.path.join(self._base_dir, 'labeled.list')
        else:
            list_path = os.path.join(self._base_dir, 'val.list')
        
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
        if num is not None and self.split == 'train':
            self.sample_list = self.sample_list[:num]

        # Since each image will be split into 4 patches (2x2), expand index list
        self.patch_per_image = (1024 // patch_size) ** 2  # 4 for 512x512 patches in 1024x1024
        self.total_patches = len(self.sample_list) * self.patch_per_image
        print("Total {} samples -> {} patches".format(len(self.sample_list), self.total_patches))

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Map global index to image index and patch index
        image_idx = idx // self.patch_per_image
        patch_idx = idx % self.patch_per_image

        case = self.sample_list[image_idx]
        h5f = h5py.File(os.path.join(self._base_dir, "data", case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Compute patch coordinates
        ps = self.patch_size
        row = patch_idx // (1024 // ps)
        col = patch_idx % (1024 // ps)
        x_start, y_start = row * ps, col * ps

        image_patch = image[x_start:x_start+ps, y_start:y_start+ps]
        label_patch = label[x_start:x_start+ps, y_start:y_start+ps]

        sample = {'image': image_patch, 'label': label_patch}

        if self.transform:
            sample = self.transform(sample)
            
        sample['image'] = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)  # (1, H, W)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.uint8))  # (H, W)

        return sample

class Unlabeled_CryoET_Patch(Dataset):
    def __init__(self, base_dir=None, num=None, transform=None, patch_size=512):
        self._base_dir = base_dir
        self.transform = transform
        self.patch_size = patch_size

        with open(os.path.join(self._base_dir, 'unlabeled.list'), 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.sample_list = self.sample_list[:num]

        self.patch_per_image = (1024 // patch_size) ** 2  # 4 patches for 512x512
        self.total_patches = len(self.sample_list) * self.patch_per_image
        print("Total {} images -> {} patches".format(len(self.sample_list), self.total_patches))

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Map global index to image and patch position
        image_idx = idx // self.patch_per_image
        patch_idx = idx % self.patch_per_image

        case = self.sample_list[image_idx]
        h5f = h5py.File(os.path.join(self._base_dir, 'data', case), 'r')
        image = h5f['image'][:]  # Expected shape: (C, H, W)
        label = h5f['label'][:]  # Optional even if "unlabeled"
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Compute patch coordinates
        ps = self.patch_size
        row = patch_idx // (1024 // ps)
        col = patch_idx % (1024 // ps)
        x_start = row * ps
        y_start = col * ps

        image_patch = image[x_start:x_start+ps, y_start:y_start+ps]
        label_patch = label[x_start:x_start+ps, y_start:y_start+ps]

        sample = {'image': image_patch, 'label': label_patch}

        if self.transform:
            sample = self.transform(sample)
            
        sample['image'] = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0)  # (1, H, W)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.uint8))  # (H, W)

        return sample

class CryoET(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/labeled.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class Unlabeled_CryoET(Dataset):
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        with open(self._base_dir + '/unlabeled.list', 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class CryoETw3Din(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/labeled.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].repeat(3, 1, 1)
        return sample

class Unlabeled_CryoETw3Din(Dataset):
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        with open(self._base_dir + '/unlabeled.list', 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].repeat(3, 1, 1)
        return sample
    
class CryoET_Patchw3Din(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, patch_size=512):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.sample_list = []

        if self.split == 'train':
            list_path = os.path.join(self._base_dir, 'labeled.list')
        else:
            list_path = os.path.join(self._base_dir, 'val.list')
        
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
        if num is not None and self.split == 'train':
            self.sample_list = self.sample_list[:num]

        # Since each image will be split into 4 patches (2x2), expand index list
        self.patch_per_image = (1024 // patch_size) ** 2  # 4 for 512x512 patches in 1024x1024
        self.total_patches = len(self.sample_list) * self.patch_per_image
        print("Total {} samples -> {} patches".format(len(self.sample_list), self.total_patches))

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Map global index to image index and patch index
        image_idx = idx // self.patch_per_image
        patch_idx = idx % self.patch_per_image

        case = self.sample_list[image_idx]
        h5f = h5py.File(os.path.join(self._base_dir, "data", case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Compute patch coordinates
        ps = self.patch_size
        row = patch_idx // (1024 // ps)
        col = patch_idx % (1024 // ps)
        x_start, y_start = row * ps, col * ps

        image_patch = image[x_start:x_start+ps, y_start:y_start+ps]
        label_patch = label[x_start:x_start+ps, y_start:y_start+ps]

        sample = {'image': image_patch, 'label': label_patch}

        if self.transform:
            sample = self.transform(sample)
            
        sample['image'] = torch.from_numpy(sample['image'].astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)  # (1, H, W)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.uint8))  # (H, W)

        return sample

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.resize = RandomCrop2D(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        sample = {'image': image, 'label': label}
        return sample
    
class RandomGeneratorWithRandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.resize = RandomCrop2D(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        sample = {'image': image, 'label': label}
        sample = self.resize(sample)
        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}

class Resize2D(object):
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple): Desired output size as (height, width).
        """
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Ensure image and label are 2D
        assert image.ndim == 2, f"Expected 2D image, got shape {image.shape}"
        assert label.ndim == 2, f"Expected 2D label, got shape {label.shape}"

        # Ensure binary label before resizing
        label = label.astype(bool)

        # Resize
        image_resized = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0, preserve_range=True).astype(np.uint8)
        label_resized = sk_trans.resize(label, self.output_size, order=0, preserve_range=True).astype(np.uint8)

        # Check binary mask
        assert np.max(label_resized) == 1 and np.min(label_resized) == 0
        assert np.unique(label_resized).shape[0] == 2

        return {'image': image_resized, 'label': label_resized}
    
class Resize3D(object):
    
    def __init__(self, output_size):
        # output_size should be a tuple (new_w, new_h, new_d)
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Ensure the image and label are 3D (w, h, d)
        assert len(image.shape) == 3, "Input image should be 3D (w, h, d)"
        assert len(label.shape) == 3, "Input label should be 3D (w, h, d)"
        
        (w, h, d) = image.shape
        label = label.astype(bool)

        # Resize image and label using skimage.transform.resize
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)  # Keep binary labels
        
        # Assert that the resized label is still binary (0 or 1)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}
    
class CenterCrop2D(object):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size  # (H, W)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[-2:]  # supports (H, W) or (C, H, W)
        crop_h, crop_w = self.output_size

        # Pad if needed
        pad_h = max((crop_h - h + 1) // 2, 0)
        pad_w = max((crop_w - w + 1) // 2, 0)

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w)], mode='constant', constant_values=0)
            label = np.pad(label, [(pad_h, pad_h), (pad_w, pad_w)], mode='constant', constant_values=0)
            h, w = image.shape[-2:]

        # Compute top-left corner of crop
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        image = image[top:top + crop_h, left:left + crop_w]
        label = label[top:top + crop_h, left:left + crop_w]
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        return {'image': image, 'label': label}
    
class RandomCrop2D(object):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size  # (H, W)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[-2:]  # handles (H, W) or (C, H, W)
        crop_h, crop_w = self.output_size

        # Pad if crop size is larger than the image
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, [(pad_h // 2, pad_h - pad_h // 2), 
                                   (pad_w // 2, pad_w - pad_w // 2)], 
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pad_h // 2, pad_h - pad_h // 2), 
                                   (pad_w // 2, pad_w - pad_w // 2)], 
                           mode='constant', constant_values=0)
            h, w = image.shape

        # Random top-left corner
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

        image = image[top:top + crop_h, left:left + crop_w]
        label = label[top:top + crop_h, left:left + crop_w]

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(label.astype(np.uint8))  # (H, W)

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        label = sample['label'].astype(np.int16)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
