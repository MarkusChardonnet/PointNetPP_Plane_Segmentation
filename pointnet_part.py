import torch
import torch.nn as nn
import torch.nn.functional as F
import grouping_sampling_part as gs
import copy

#Set Abstraction layer
class SA(nn.Module):

    def __init__(self, n_sample, radius, layer_sizes, in_channels, group_size):
        super(SA, self).__init__()
        self.n_sample = n_sample
        self.radius = radius
        self.group_size = group_size
        self.width = len(layer_sizes)

        self.conv = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        in_fea = in_channels
        for i in range(self.width):
            self.conv.append(nn.Conv2d(in_fea, layer_sizes[i], 1))
            self.batchnorm.append(nn.BatchNorm2d(layer_sizes[i]))
            in_fea = layer_sizes[i]
 


    def forward(self, pts_co, pts_fea, distance = None, sample = None):
        #pts_co of size [b,n,3]        
        #pts_fea of size [b,n,c]

        new_pts_co, new_pts_fea, dist, samples = gs.sampling_grouping(pts_co, pts_fea, self.n_sample, self.radius, self.group_size, distance, sample)

        new_pts_fea = new_pts_fea.permute(0,3,1,2)
        for i in range(self.width):
            new_pts_fea = F.relu(self.batchnorm[i](self.conv[i](new_pts_fea)))
        new_pts_fea = new_pts_fea.permute(0,2,3,1)
        new_pts_fea, m = torch.max(new_pts_fea, dim=2)

        return new_pts_co, new_pts_fea, dist, samples #[b,m,3], [b,m,c+3], [b,n,m], [b,m]

#Feature propagation layer
class FP(nn.Module):
    def __init__(self, layer_sizes, in_channels, dropout=None):
        super(FP, self).__init__()
        self.width = len(layer_sizes)
        in_fea = in_channels

        self.conv = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(self.width):
            self.conv.append(nn.Conv1d(in_fea, layer_sizes[i], 1))
            self.batchnorm.append(nn.BatchNorm1d(layer_sizes[i]))
            in_fea = layer_sizes[i]

        self.drop = nn.Dropout()
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = torch.zeros(self.width, dtype=torch.bool)

    def forward(self, pts_co_small, pts_fea_small, pts_co_big, pts_fea_big, distance=None):
        # pts_co_small [b,n,3]
        # pts_co_big [b,m,3]
        # pts_fea_small [b,n,c_]
        # pts_fea_big [b,m,c]
        # distance [b,n,m] or None

        device = pts_co_small.device

        b = pts_co_small.shape[0]
        n = pts_co_small.shape[1]
        m = pts_fea_big.shape[1]
        c = pts_fea_big.shape[2]

        # features propagated from higher level
        features = torch.zeros([b, n, c]).to(device)
        if pts_co_big is None:
            features = pts_fea_big.view(b, 1, c).repeat(1, n, 1)
        else:
            if distance is None:
                distance = gs.compute_distance(pts_co_small, pts_co_big)

            sorted_dist, neighbors = torch.sort(distance, axis=2)
            is_big = sorted_dist[:, :, 0] == 0

            single_features = gs.pts_index_query(pts_fea_big, neighbors[:, :, 0])

            weights = (1 / sorted_dist[:, :, 0:3]).to(device).view(b, n, 3, 1).repeat([1, 1, 1, c])
            multiple_features = torch.sum(gs.pts_index_query(pts_fea_big, neighbors[:, :, 0:3]) * weights, dim=2) / torch.sum(weights, dim=2)

            features = multiple_features
            features[is_big] = single_features[is_big]

        # concatenated with features from current level
        x = None
        if pts_fea_small is not None:
            x = torch.cat((pts_fea_small, features), dim=2).to(device)
        else:
            x = features.to(device)

        x = x.permute(0, 2, 1)
        for i in range(self.width):
            x = F.relu(self.batchnorm[i](self.conv[i](x)))
            if self.dropout[i]:
                x = self.drop(x)
        x = x.permute(0, 2, 1)

        return x  # [b,n,c+c_]

#Whole architecture
class PointNet2(nn.Module):
    def __init__(self, labels):
        super(PointNet2, self).__init__()
        self.bn = nn.BatchNorm1d(3)
        self.sa1 = SA(512, 0.2, [64,64,128], 6, 32)
        self.sa2 = SA(128, 0.4, [128,128,256], 131, 64)
        self.sa3 = SA(64, None, [256,512,1024], 256, None)

        self.fp3 = FP([256,256],1280)
        self.fp2 = FP([256,128],384)
        self.fp1 = FP([128,128,128,128],128, dropout=[False, False, True, True])
        self.singleconv = nn.Conv1d(128,labels,1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, pts_co):
        #pts_co [b,n,3]

        device = pts_co.device
        pts_fea = pts_co.clone()

        pts_co1, pts_fea1, distance1, sample1 = self.sa1(pts_co, pts_fea)
        pts_co2, pts_fea2, distance2, sample2 = self.sa2(pts_co1, pts_fea1, distance1, sample1)
        pts_co3, pts_fea3, distance3, sample3 = self.sa3(pts_co2, pts_fea2, distance2, sample2)

        pts_fea2 = self.fp3(pts_co2, pts_fea2, pts_co3, pts_fea3, distance3)
        pts_fea1 = self.fp2(pts_co1, pts_fea1, pts_co2, pts_fea2, distance2)
        seg = self.fp1(pts_co, None, pts_co1, pts_fea1, distance1).to(device)

        seg = seg.permute(0,2,1)
        seg = self.softmax(self.singleconv(seg))
        seg = seg.permute(0,2,1)

        return seg #[b,n,labels]