import os
import random

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchvision.transforms import ColorJitter
import glob
from tqdm import tqdm

def get_files(PATH: str | list[str]) -> list[str]:
    """Get all files in a directory recursively."""
    file_list: list[str] = []
    if isinstance(PATH, str):
        for filepath, dirnames, filenames in os.walk(PATH):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
    elif isinstance(PATH, list):
        for path in PATH:
            for filepath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_list.append(os.path.join(filepath, filename))
    return file_list

class GOSrandomAffine:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        self.transform = transforms.RandomAffine(degrees=30, translate=(0, 0.25), scale=(0.8, 1.2), shear=15, fill=0)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < self.prob:
            image, gt = sample['image'], sample['gt']
            
            # Apply random perspective transform to both image and ground truth
            im = torch.cat([image,gt],dim=0)
            im = self.transform(im)
            # gt = self.transform(gt)
            image = im[:3,:,:]
            gt = im[3:,:,:]
            
            sample['image'] = image
            sample['gt'] = gt
        
        return sample

class GOSrandomPerspective:
    def __init__(self, prob: float = 0.5, distortion_scale: float = 0.5, p: float = 1.0) -> None:
        self.prob = prob
        self.transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < self.prob:
            image, gt = sample['image'], sample['gt']
            
            # Apply random perspective transform to both image and ground truth
            im = torch.cat([image,gt],dim=0)
            im = self.transform(im)
            # gt = self.transform(gt)
            image = im[:3,:,:]
            gt = im[3:,:,:]
            
            sample['image'] = image
            sample['gt'] = gt
        
        return sample

class GOSGaussianNoise:
    def __init__(self, max_std: float = 0.2, prob: float = 0.5) -> None:
        super().__init__()
        self.max_std = max_std
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < self.prob:
            image =  sample['image']
            noise = torch.randn(image.shape) * torch.rand(1) * self.max_std
            noisy_img_tensor = image + noise
            sample['image'] = noisy_img_tensor
        return sample

def rotate_and_crop(img: torch.Tensor, angle: float) -> torch.Tensor:
    rotated_img = transforms.functional.rotate(img, angle)
    _, h_orig, w_orig = img.shape
    theta = abs(torch.tensor(np.radians(angle % 180)))
    if theta > torch.pi/2:
        theta = torch.pi - theta
    new_h = int(h_orig /(torch.cos(theta) + torch.sin(theta)))
    new_w = new_h
    top = (h_orig - new_h) // 2
    left = (h_orig - new_w) // 2
    cropped_img = rotated_img[:, top:top+new_h, left:left+new_w]
    cropped_img = transforms.functional.resize(cropped_img, (h_orig, w_orig))
    return cropped_img

class GOSrandomRotation:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        if random.random() < self.prob:
            image, gt, depth =  sample['image'], sample['gt'], sample['depth']
            depth_large = sample['depth_large']
            random_angle = np.random.randint(-30, 30)
            # print(image.shape)
            image = rotate_and_crop(image,random_angle)
            gt = rotate_and_crop(gt,random_angle)
            depth = rotate_and_crop(depth,random_angle)
            depth_large = rotate_and_crop(depth_large,random_angle)
            sample['image'] = image
            sample['gt'] = gt
            sample['depth'] = depth
            sample['depth_large'] = depth_large
        return sample

class GOSColorEnhance:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < self.prob:
            # Convert tensor to numpy array (H, W, C) format for OpenCV
            image = sample['image'] * 255.0
            image = np.uint8(image.permute(1, 2, 0).cpu().numpy())

            # Apply brightness adjustment
            bright_intensity = random.randint(5, 15) / 10.0
            image = cv2.convertScaleAbs(image, alpha=bright_intensity, beta=0)

            # Apply contrast adjustment
            contrast_intensity = random.randint(5, 15) / 10.0
            image = cv2.convertScaleAbs(image, alpha=contrast_intensity, beta=128 * (1 - contrast_intensity))

            # Apply color (saturation) adjustment
            color_intensity = random.randint(0, 20) / 10.0
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * color_intensity, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Apply sharpness adjustment
            sharp_intensity = random.randint(0, 30) / 10.0
            if sharp_intensity > 1.0:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * (sharp_intensity - 1.0) / 2.0
                kernel[1, 1] = 1 + kernel[1, 1]
                image = cv2.filter2D(image, -1, kernel)
                image = np.clip(image, 0, 255).astype(np.uint8)

            # Convert back to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            sample['image'] = image

        return sample

class GOSColorJitter:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        self.ColorJitter = ColorJitter(0.1, 0.1, 0.1, 0.1)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        if random.random() < self.prob:
            image = sample['image']
            image = self.ColorJitter(image)
            sample['image'] = image
        
        return sample
    
class GOSRandomUPCrop(object):
    def __init__(self, prob=0.5, border=30):
        self.prob = prob
        self.border = border
    def __call__(self,sample):
        # flag = 1
        if random.random() < self.prob:
            image = sample['image']
            gt = sample['gt']
            depth = sample['depth']
            depth_large = sample['depth_large']
            
            image_height, image_width = image.shape[-2], image.shape[-1]
            
            crop_win_width = torch.randint(image_width - self.border, image_width, (1,)).item()
            crop_win_height = torch.randint(image_height - self.border, image_height, (1,)).item()

            x_start = (image_width - crop_win_width) // 2
            y_start = (image_height - crop_win_height) // 2
            x_end = x_start + crop_win_width
            y_end = y_start + crop_win_height

            cropped_image = image[..., y_start:y_end, x_start:x_end]
            cropped_gt = gt[..., y_start:y_end, x_start:x_end]
            cropped_depth = depth[..., y_start:y_end, x_start:x_end]
            cropped_depth_large = depth_large[..., y_start:y_end, x_start:x_end]
            image_cropped = F.interpolate(cropped_image[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            gt_cropped = F.interpolate(cropped_gt[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            depth_cropped = F.interpolate(cropped_depth[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            cropped_depth_large = F.interpolate(cropped_depth_large[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            sample['image'] = image_cropped
            sample['gt'] = gt_cropped
            sample['depth'] = depth_cropped
            sample['depth_large'] = cropped_depth_large
        return sample



class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406,0], std=[0.229,0.224,0.225,1.0]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        image = sample['image']
        image = normalize(image,self.mean,self.std)
        sample['image'] = image
        return sample

class GOSMAXNormalize(object):
    def __init__(self):
        pass
    def __call__(self,sample):
        image =  sample['image']
        image = (image-image.min()) / (image.max() - image.min())
        sample['image'] = image
        return sample

class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        # random horizontal flip
        if random.random() <= self.prob:
            image, gt =  sample['image'], sample['gt']
            depth, depth_large = sample['depth'],sample['depth_large']
            image = torch.flip(image,dims=[2])
            gt = torch.flip(gt,dims=[2])
            depth = torch.flip(depth,dims=[2])
            depth_large = torch.flip(depth_large,dims=[2])
            sample['image'] = image
            sample['gt'] = gt
            sample['depth'] = depth
            sample['depth_large'] = depth_large
        return sample

class GOSRandomimg2Grayedge(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            gt = sample['gt'][None,...]

            kernel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kernel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


            edges_x = torch.nn.functional.conv2d(gt, kernel_x, padding=1)
            edges_y = torch.nn.functional.conv2d(gt, kernel_y, padding=1)


            edges = torch.sqrt(edges_x ** 2 + edges_y ** 2).repeat(1,3,1,1)
            sample['image'] = edges[0]
        return sample

class GOSRandombackground2same(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            im = sample['image']
            gt = sample['gt']
            objects = im*gt
            objects_mean_rgb = objects.mean(dim=(1,2),keepdim=True)
            same_back_ground = objects_mean_rgb*(1-gt)
            sample['image'] = objects + same_back_ground
        return sample
    
class GOSRandombackground2edgesame(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            im = sample['image']
            gt = sample['gt']
            edges = abs(gt[None,...]-F.avg_pool2d(gt[None,...],kernel_size=31,stride=1,padding=15))[0]
            edges = (edges - edges.min()) / (edges.max() - edges.min())
            avg_im = F.avg_pool2d(im[None,...],kernel_size=31,stride=1,padding=15)[0]
            sample['image'] = im * (1-edges) + avg_im * edges
        return sample

class GOSRandomGray(object):
    def __init__(self,prob=0.5):
        self.prob = prob
        self.transform = transforms.Grayscale(num_output_channels=3)
    def __call__(self,sample):
        if random.random() <= self.prob:
            image = sample['image']
            image = self.transform(image)
            sample['image'] = image
        return sample

class GOSTorchRandomCrop(object):
    def __init__(self, prob=0.5, border=30):
        self.prob = prob
        self.border = border

    def __call__(self, sample):
        if random.random() < self.prob:
            image = sample['image']
            gt = sample['gt']


            image_height, image_width = image.shape[1], image.shape[2]

            crop_win_width = np.random.randint(image_width - self.border, image_width)
            crop_win_height = np.random.randint(image_height - self.border, image_height)


            top = (image_height - crop_win_height) // 2
            left = (image_width - crop_win_width) // 2
            bottom = top + crop_win_height
            right = left + crop_win_width

            image_cropped = image[:, top:bottom, left:right]
            gt_cropped = gt[:, top:bottom, left:right]

            sample['image'] = image_cropped
            sample['gt'] = gt_cropped

        return sample

class DISDataset(Dataset):
    """Dataset for Dichotomous Image Segmentation (DIS)."""

    def __init__(self,root,transform=[],chached=False,size=[224,224],stoi=None,istrain=0,use_gt=True):
        self.istrain = istrain
        self.imlists = get_files(root)
        self.transforms = transforms.Compose(transform)
        self.chached = chached
        self.size = size
        self.use_gt = use_gt
        if use_gt:
            if stoi is None:
                label_chache = []
                for i in range(len(self.imlists)):
                    label_chache.append(''.join(self.imlists[i].split('/')[-1].split('#')[0:3]))
                    label_chache = sorted(list(set(label_chache)))
                self.stoi = { ch:i for i,ch in enumerate(label_chache) }
            else:
                self.stoi = stoi
        else:
            self.stoi = {"0":0}
        if self.chached:
            self.imlists_chache = []
            self.gt_chache = []
            self.raw_size = []
            for im in tqdm(range(len(self.imlists))):
                tmpimg = cv2.cvtColor(cv2.imread(self.imlists[im]),cv2.COLOR_BGR2RGB)
                self.raw_size.append(tmpimg.shape)

                tmpimg = F.interpolate(torch.from_numpy(tmpimg).permute(2,0,1)[None,...],size=size,mode='bilinear',align_corners=True)[0]
                self.imlists_chache.append(tmpimg)
                if use_gt:
                    tmplabel = cv2.cvtColor(cv2.imread(self.imlists[im].replace('/images','/masks').replace('.jpg','.png')),cv2.COLOR_BGR2GRAY)
                    tmplabel = F.interpolate(torch.from_numpy(tmplabel)[None,None,...],size=size,mode='bilinear', align_corners=True)[0][0]
                else:
                    tmplabel = torch.zeros([1,size[0],size[1]])
                self.gt_chache.append(tmplabel)

    def __getitem__(self, index):
        
        if self.chached:
            im = self.imlists_chache[index]
            gt = self.gt_chache[index]
            raw_size = self.raw_size[index][:2]
            one_hot_label = torch.zeros([len(self.stoi)])
            one_hot_label[self.stoi[''.join(self.imlists[index].split('/')[-1].split('#')[0:3])]] = 1
            label =  one_hot_label
        else:
            im = cv2.cvtColor(cv2.imread(self.imlists[index]),cv2.COLOR_BGR2RGB)
            raw_size = im.shape[:2]
            im = F.interpolate(torch.from_numpy(im).permute(2,0,1)[None,...],size=self.size,mode='bilinear',align_corners=True)[0]
            if self.use_gt:
                gt = cv2.cvtColor(cv2.imread(self.imlists[index].replace('/images','/masks').replace('.jpg','.png')),cv2.COLOR_BGR2GRAY)
                gt = F.interpolate(torch.from_numpy(gt)[None,None,...],size=self.size,mode='nearest')[0][0]
            else:
                gt = torch.zeros([1,self.size[0],self.size[1]])
            one_hot_label = torch.zeros([len(self.stoi)])
            if self.use_gt:
                one_hot_label[self.stoi[''.join(self.imlists[index].split('/')[-1].split('#')[0:3])]] = 1
            label =  one_hot_label
            random_num = random.random()
            if not self.istrain:
                random_num=1
            if random_num > 0.66:
                depth = cv2.cvtColor(cv2.imread(self.imlists[index].replace('/images','/depth_large')),cv2.COLOR_BGR2GRAY)
            elif random_num > 0.33:
                depth = cv2.cvtColor(cv2.imread(self.imlists[index].replace('/images','/depth_base')),cv2.COLOR_BGR2GRAY)
            else:
                depth = cv2.cvtColor(cv2.imread(self.imlists[index].replace('/images','/depth_small')),cv2.COLOR_BGR2GRAY)
            depth = F.interpolate(torch.from_numpy(depth)[None,None,...],size=self.size,mode='bilinear',align_corners=True)[0]
            if self.istrain:
                large_depth = cv2.cvtColor(cv2.imread(self.imlists[index].replace('/images','/depth_large_1024')),cv2.COLOR_BGR2GRAY)
                large_depth = F.interpolate(torch.from_numpy(large_depth)[None,None,...],size=self.size,mode='bilinear',align_corners=True)[0]
                large_depth = torch.divide(large_depth,255.0)
        # depth = torch.zeros_like(im)
        im = torch.divide(im,255.0)
        gt = torch.divide(gt,255.0)
        depth = torch.divide(depth,255.0)
        sample = {
            'image_name':self.imlists[index],
            'image_size':raw_size,
            "image": im.float(),
            "gt": gt.float()[None,...],
            "depth": depth.float(),
            "depth_large": large_depth.float() if self.istrain else depth.float(),
            "label": label,
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
    def __len__(self):
        return self.imlists.__len__()
    
def build_dataset(is_train,args):
    if is_train:
        train_data_path = [
            args.data_path+'/DIS-TR/images',
                           ]
        return DISDataset(train_data_path,transform=[
            # Random flip
            GOSRandomHFlip(0.5),
            # GOSrandomPerspective(0.5),
            # GOSrandomAffine(0.5),
            GOSrandomRotation(0.5),
            # GOSRandombackground2same(1),
            # GOSRandombackground2edgesame(1),
            # GOSRandomimg2Grayedge(0.25),
            # Color augmentation
            # GOSColorJitter(0.5),
            GOSColorEnhance(0.5),
            GOSRandomGray(0.25),
            # Random crop
            GOSRandomUPCrop(0.5),
            # GOSTorchRandomCrop(0.5),
            # Normalization
            GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Noise
            # GOSGaussianNoise(0.1,prob=0.5),
            ],chached=args.chached,size=[args.input_size,args.input_size],istrain=True)
    else:
        valid_data_path = args.data_path+'/DIS-VD/images'
        return DISDataset(valid_data_path,transform=[GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],chached=args.chached,size=[args.input_size,args.input_size])
def keep_n_files(directory, n=3):
    files = [(file_path, os.path.getmtime(file_path)) for file_path in glob.glob(os.path.join(directory, '*'))]
    
    files.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, _ in files[n:]:
        os.remove(file_path)