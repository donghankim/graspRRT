import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette
import math, pdb


# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        self.root_dir = img_dir
        self.rgb_dir = self.root_dir + "rgb/"
        self.gt_dir = self.root_dir + "gt/"
        self.depth_dir = self.root_dir + "depth/"
        self.pred_dir = self.root_dir + "pred"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean_rgb, std = std_rgb)
        ])

        self.dataset_cnt = 0
        for path in os.listdir(self.rgb_dir):
            if os.path.isfile(os.path.join(self.rgb_dir, path)):
                self.dataset_cnt+= 1

    def __len__(self):
        return self.dataset_cnt

    def __getitem__(self, idx):
        rgb_path = f"{self.rgb_dir}{idx}_rgb.png"
        gt_path = f"{self.gt_dir}{idx}_gt.png"
        
        rgb_img = self.transform(image.read_rgb(rgb_path))
        gt_mask = torch.LongTensor(image.read_mask(gt_path))
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample



class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
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


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")
                
                # added to save corresponding rgb image
                rgb_img = data[i].cpu().numpy()
                image.write_rgb(rgb_img, f"{dump_dir}/{test_ID}_rgb.png")


def iou(prediction, target):
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    class_num = prediction.shape[1]
    batch_ious = list()
    for batch_id in range(batch_num):
        class_ious = list()
        for class_id in range(1, class_num):  # class 0 is background
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0: # skip the occluded object
                continue
            intersection = (mask_pred * mask_target).sum()
            union = (mask_pred + mask_target).sum() - intersection
            class_ious.append(float(intersection) / float(union))
        batch_ious.append(np.mean(class_ious))
    return batch_ious


def run(model, loader, criterion, is_train=False, optimizer=None):
    model.train(is_train)
    mean_epoch_loss, mean_iou = 0.0, 0.0
    n_iters = math.ceil(len(loader.dataset)/loader.batch_size)
    data_iter = iter(loader)

    for _ in range(n_iters): 
        batch = next(data_iter)
        img, gt = batch['input'], batch['target']
        X = img.to(device)
        Y = gt.to(device)
        Y_P = model.forward(X)
        
        optimizer.zero_grad()
        loss = criterion(Y_P, Y)
        loss.backward()
        optimizer.step()
        
        mean_iou += sum(iou(Y_P, Y))
        mean_epoch_loss += loss.item()
        
    mean_iou /= len(loader.dataset)
    mean_epoch_loss /= len(loader.dataset)
    
    return mean_epoch_loss, mean_iou


def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)


if __name__ == "__main__":
    seed(0)
    torch.manual_seed(0)


    # Check if GPU is being detected (added support for m-series mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    dataset = RGBDataset("./dataset/")
    train_size = int(len(dataset)*0.9)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    batch_size = 10
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

    model = miniUNet(3, 4)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025)

    model_path = "checkpoint_multi.pth.tar"
    epoch, max_epochs = 1, 15
    best_miou = float('-inf')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = run(model, train_loader, criterion, is_train = True, optimizer = optimizer)
        test_loss, test_miou = run(model, test_loader, criterion, is_train = False, optimizer = optimizer)
 
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Test loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou 
            save_chkpt(model, epoch, test_miou, model_path)
        epoch += 1
        lr_scheduler.step()
    
    if best_miou >= 0.95:
        model, epoch, best_miou = load_chkpt(model, model_path, device)
        save_prediction(model, test_loader, dataset.pred_dir, device, batch_size)
        print("predictions saved!")

