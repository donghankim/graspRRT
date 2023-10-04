import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


class RGBDataset(Dataset):
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        self.root_dir = img_dir
        self.rgb_dir = self.root_dir + "rgb/"
        self.gt_dir = self.root_dir + "gt/"
        self.depth_dir = self.root_dir + "depth/"
        self.pred_dir = self.root_dir + "pred"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = RGBDataset.mean_rgb, std = RGBDataset.std_rgb)
        ])

        self.dataset_cnt = 0
        for path in os.listdir(self.rgb_dir):
            if os.path.isfile(os.path.join(self.rgb_dir, path)):
                self.dataset_cnt+= 1

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        return self.dataset_cnt

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        rgb_path = f"{self.rgb_dir}{idx}_rgb.png"
        gt_path = f"{self.gt_dir}{idx}_gt.png"
        
        rgb_img = self.transform(image.read_rgb(rgb_path))
        gt_mask = torch.LongTensor(image.read_mask(gt_path))
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample


class ActionRegressionDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for ActionRegressionModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32
            'target': torch.Tensor (3,), torch.float32
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        Note: target: [x, y, angle] scaled to between 0 and 1.
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # ===============================================================================
        data_dict = {}
        rgb = data['rgb'].numpy()
        rgb = (rgb/np.max(rgb)).transpose((2,0,1))
        angle = data['angle'].numpy().item()
        center = data['center_point'].numpy()
        nx = center[0]/rgb.shape[-1]
        ny = center[1]/rgb.shape[-2]
        na = angle/180
        target = np.array([nx,ny,na], dtype = np.float32)
        
        data_dict['input'] = torch.from_numpy(rgb).type(torch.float32)
        data_dict['target'] = torch.from_numpy(target).type(torch.float32)
        return data_dict
        # ===============================================================================


class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        input_img = data['rgb'].numpy()
        kp = data['center_point'].numpy()
        angle = data['angle'].numpy().item()
        bins = np.arange(0, 180, 22.5)
        r_idx = np.digitize(angle, bins)
        l_idx = r_idx-1
        if r_idx == 8:
            bin_angle = 0 if abs(180 - angle) < abs(angle - bins[l_idx]) else bins[l_idx]
        else:
            bin_angle = bins[r_idx] if abs(bins[r_idx] - angle) < abs(angle - bins[l_idx]) else bins[l_idx]
        
        kps = KeypointsOnImage([Keypoint(x = kp[0], y = kp[1])], shape = input_img.shape[:2])
        seq = iaa.Sequential([iaa.Affine(rotate = -bin_angle)])
        rot_img, rot_kps = seq(image = input_img, keypoints = kps)
        rot_img = (rot_img/np.max(rot_img)).transpose((2,0,1))
        target_img = get_gaussian_scoremap((input_img.shape[0], input_img.shape[1]), rot_kps[0].xy)
        data = dict(input = torch.from_numpy(rot_img).type(torch.float32), target = torch.from_numpy(target_img).unsqueeze(0).type(torch.float32))
        return data 
