import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model.vig_pytorch.gcn_lib.torch_vertex import DyGraphConv2d


class MyModel(nn.Module):
    def __init__(self,a1,b1):
        super(MyModel, self).__init__()
        self.a1 = a1
        self.b1 = b1
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ConvTranspose2d(64, 5, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(5, 5, kernel_size=1)  
        )
        self.linear = nn.Linear(192, 193) 

        self.upper1 = nn.Conv2d(64,5,1,1) 
        self.upper2 = nn.Conv2d(128,5,1,1)

        self.upper = nn.Conv2d(64,128,1,1)
        self.ssim_loss = SSIMLoss()

        self.gnn1 =  DyGraphConv2d(64,64)
        self.gnn2 =  DyGraphConv2d(128,64)
    def forward(self, x):
        x1 = self.encoder(x)  
        x1g = self.gnn1 (x1) 
        
        x2 = self.middle(x1)  
        x2g = self.gnn2 (x2)
        
        x = self.decoder(x2)  
        x = self.linear(x) 
        
        
        return x, self.upper1(x1), x2, F.adaptive_avg_pool2d(x1g, (4, 48)) ,x2g

    def compute_loss(self, output, target):
        output, x1, x2 ,x1g,x2g= output

        loss_main = F.mse_loss(output, target)
        
        # 辅助损失1：编码器输出与目标下采样后的MSE损失
        target_downsampled = F.interpolate(target, size=x1.shape[2:])
        

        loss_aux1 = self.ssim_loss(x1, target_downsampled)
        
        loss_g = F.mse_loss(x1g, x2g)
        loss_total = loss_main + self.a1 * loss_g  + self.b1  * loss_aux1  # 2.62
        return loss_total

################


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)