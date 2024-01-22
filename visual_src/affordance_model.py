from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque
from train import get_finger_points
import pdb


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

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
        data = self.raw_dataset[idx]
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


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        device = self.device
        batch_obs = np.empty((8,) + rgb_obs.shape)
        for i in range(8):
            rot_val = i*22.5
            seq = iaa.Sequential([iaa.Affine(rotate = -rot_val)])
            rot_img, _ = seq(image = rgb_obs, keypoints = None)
            rot_img = (rot_img/np.max(rot_img))
            batch_obs[i] = rot_img

        batch_tensor = torch.from_numpy(batch_obs.transpose((0,3,1,2))).type(torch.float32).to(device)
        batch_res = self.predict(batch_tensor)
        affordance_map = torch.clip(batch_res, 0, 1).detach()
        
        for max_coord in list(self.past_actions):
            angle_bin = max_coord[0]
            # affordance_map[angle_bin] = 0
            affordance_map[max_coord] = 0

        res_idx = unravel_index(torch.argmax(affordance_map).item(), affordance_map.shape)
        self.past_actions.append(res_idx)

        target_coord = (res_idx[-1], res_idx[-2])
        angle = 22.5*res_idx[0]
        rgb_img = (batch_obs[res_idx[0]]*255.0).astype(np.uint8)
        seq = iaa.Sequential([iaa.Affine(rotate = angle)])
        kps = KeypointsOnImage([Keypoint(x = target_coord[0], y = target_coord[1])], shape = rgb_obs.shape[:2])
        _, rc = seq(image = rgb_img, keypoints = kps)
        coord = (min(rc[0].x_int, rgb_obs.shape[1]-1), min(rc[0].y_int, rgb_obs.shape[0]-1)) # rare, out-of-bounds only happens with masterChefCan
        
        cmap = cm.get_cmap('viridis')
        canvas = np.empty((rgb_obs.shape[0]*8, rgb_obs.shape[1]*2, 3), dtype = np.uint8)
        for i in range(8):
            rgb_img = (batch_obs[i]*255.0).astype(np.uint8)
            target_img = affordance_map[i].cpu().numpy().transpose(1,2,0)
            target_img = (cmap(target_img[:,:,0])[...,:3]*255.0).astype(np.uint8)
            if i == res_idx[0]:
                draw_grasp(rgb_img, target_coord, 0)
            
            curr_img = np.concatenate([rgb_img, target_img], axis=1)
            curr_img[-1] = 127
            canvas[i*128:(i+1)*128] = curr_img

        r1, r2 = np.vsplit(canvas, 2)
        vis_img = np.concatenate([r1, r2], axis = 1)
        return coord, angle, vis_img

