import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# image segmentation model
class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.conv1= nn.Conv2d(3,16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(384, 128, kernel_size = 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(192, 64, kernel_size = 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(96, 32, kernel_size = 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(48, 16, kernel_size = 3, padding = 1)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.Conv2d(16, self.n_classes, kernel_size = 1, padding = 0)
        self.bn10 = nn.BatchNorm2d(self.n_classes)

        self.pool = nn.MaxPool2d(2,2)
        self.dp = nn.Dropout(0.1)
        # ===============================================================================

    def forward(self, x):
        e1 = self.bn1(F.relu(self.conv1(x)))
        e2 = self.bn2(F.relu(self.conv2(self.pool(e1))))
        e3 = self.bn3(F.relu(self.conv3(self.pool(e2))))
        e4 = self.bn4(F.relu(self.conv4(self.pool(e3))))
        e5 = self.bn5(F.relu(self.conv5(self.pool(e4))))
        
        # decode
        _d4 = self.dp(torch.concat([e4,F.interpolate(e5, scale_factor = 2)], axis = 1))
        d4 = self.bn6(F.relu(self.conv6(_d4)))
        _d3 = self.dp(torch.concat([e3, F.interpolate(d4, scale_factor = 2)], axis = 1))
        d3 = self.bn7(F.relu(self.conv7(_d3)))
        _d2 = self.dp(torch.concat([e2, F.interpolate(d3, scale_factor = 2)], axis = 1))
        d2 = self.bn8(F.relu(self.conv8(_d2)))
        _d1 = self.dp(torch.concat([e1, F.interpolate(d2, scale_factor = 2)], axis = 1))
        d1 = self.bn9(F.relu(self.conv9(_d1)))

        output = self.bn10(F.relu(self.conv10(d1)))
        return output
        # ===============================================================================



# visual affordance model
class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
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
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
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
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
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
        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        for max_coord in list(self.past_actions):
            angle_bin = max_coord[0]
            # affordance_map[angle_bin] = 0
            affordance_map[max_coord] = 0
            # supress past actions and select next-best action

        res_idx = unravel_index(torch.argmax(affordance_map).item(), affordance_map.shape)
        self.past_actions.append(res_idx)

        target_coord = (res_idx[-1], res_idx[-2])
        angle = 22.5*res_idx[0]
        rgb_img = (batch_obs[res_idx[0]]*255.0).astype(np.uint8)
        seq = iaa.Sequential([iaa.Affine(rotate = angle)])
        kps = KeypointsOnImage([Keypoint(x = target_coord[0], y = target_coord[1])], shape = rgb_obs.shape[:2])
        _, rc = seq(image = rgb_img, keypoints = kps)
        coord = (min(rc[0].x_int, rgb_obs.shape[1]-1), min(rc[0].y_int, rgb_obs.shape[0]-1)) # rare, out-of-bounds only happens with masterChefCan
        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
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
        # ===============================================================================
        return coord, angle, vis_img


# action regression model
class ActionRegressionModel(nn.Module):
    def __init__(self, pretrained=False, out_channels=3, **kwargs):
        super().__init__()
        # load backbone model
        model = mobilenet_v3_small(pretrained=pretrained)
        # replace the last linear layer to change output dimention to 3
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        # normalize RGB input to zero mean and unit variance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        """
        Think: Why is this the same as forward 
        (comparing to AffordanceModel.predict)
        """
        return self.forward(x)

    @staticmethod
    def get_criterion():
        """
        Return the Loss object needed for training.
        """
        # TODO: complete this method
        # =============================================================================== 
        return nn.L1Loss()
        # =============================================================================== 

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """        
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
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Hint: use recover_action
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        rgb_input = (rgb_obs/np.max(rgb_obs)).transpose((2,0,1))
        rgb_tensor = torch.from_numpy(rgb_input).unsqueeze(0).type(torch.float32).to(device)
        pred = self.predict(rgb_tensor).cpu().detach().numpy()[0]
        coord, angle = recover_action(pred)
        # ===============================================================================
        # visualization
        vis_img = self.visualize(rgb_input, pred)
        return coord, angle, vis_img
