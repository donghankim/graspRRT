from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from pick_labeler import draw_grasp
import pdb


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
        data = self.raw_dataset[idx]
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


def recover_action(
        action: np.ndarray, 
        shape=(128,128)
        ) -> Tuple[Tuple[int, int], float]:
    x, y, angle = action[0], action[1], action[2]
    coord = (round(x*shape[0]), round(y*shape[1]))
    angle *= 180
    return coord, angle


class ActionRegressionModel(nn.Module):
    def __init__(self, pretrained=False, out_channels=3, **kwargs):
        super().__init__()
        model = mobilenet_v3_small(pretrained=pretrained)
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        return self.forward(x)

    @staticmethod
    def get_criterion():
        return nn.L1Loss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        vis_img = (np.moveaxis(input,0,-1).copy() * 255).astype(np.uint8)
        # target
        if target is not None:
            coord, angle = recover_action(target, shape=vis_img.shape[:2])
            draw_grasp(vis_img, coord, angle, color=(255,255,255))
        # pred
        coord, angle = recover_action(output, shape=vis_img.shape[:2])
        draw_grasp(vis_img, coord, angle, color=(0,255,0))
        return vis_img

    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        device = self.device
        rgb_input = (rgb_obs/np.max(rgb_obs)).transpose((2,0,1))
        rgb_tensor = torch.from_numpy(rgb_input).unsqueeze(0).type(torch.float32).to(device)
        pred = self.predict(rgb_tensor).cpu().detach().numpy()[0]
        coord, angle = recover_action(pred)
        
        # visualization
        vis_img = self.visualize(rgb_input, pred)
        return coord, angle, vis_img

