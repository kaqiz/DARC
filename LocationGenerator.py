import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Conv2d



def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size)



def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilated, self).__init__()
        self.dilation = dilation

    def forward(self, edge_index):
        return edge_index[:, :, :, ::self.dilation]


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.k = k
        self.dilation = dilation
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x):
        x = F.normalize(x, p=2.0, dim=1)
        edge_index = dense_knn_matrix(x, self.k * self.dilation)
        return self._dilated(edge_index)


class EdgeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv2d, self).__init__()
        self.nn = nn.Sequential(
            Conv2d(in_channels * 2, out_channels, 1, bias=True, groups=4),
            nn.ReLU(inplace=True),
        )
        for m in self.nn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class DyGraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1):
        super(DyGraphConv2d, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.gconv = EdgeConv2d(in_channels, out_channels)
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x)
        x = self.gconv(x, edge_index)
        return x.reshape(B, -1, H, W).contiguous()



class LocationGenerator(nn.Module):
    def __init__(self, a1, b1):
        super(LocationGenerator, self).__init__()
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

        self.upper1 = nn.Conv2d(64, 5, 1, 1)
        self.upper2 = nn.Conv2d(128, 5, 1, 1)

        self.upper = nn.Conv2d(64, 128, 1, 1)
        self.ssim_loss = SSIMLoss()

        self.gnn1 = DyGraphConv2d(64, 64)
        self.gnn2 = DyGraphConv2d(128, 64)

    def forward(self, x):
        x1 = self.encoder(x)
        x1g = self.gnn1(x1)

        x2 = self.middle(x1)
        x2g = self.gnn2(x2)

        x = self.decoder(x2)
        x = self.linear(x)

        return x, self.upper1(x1), x2, F.adaptive_avg_pool2d(x1g, (4, 48)), x2g

    def compute_loss(self, output, target):
        output, x1, x2, x1g, x2g = output

        loss_main = F.mse_loss(output, target)
        target_downsampled = F.interpolate(target, size=x1.shape[2:])
        loss_aux1 = self.ssim_loss(x1, target_downsampled)
        loss_g = F.mse_loss(x1g, x2g)

        loss_total = loss_main + self.a1 * loss_g + self.b1 * loss_aux1
        return loss_total
