import torch
import numpy as np


def compute_distance(pts_co, pts_co2):
    # To compute sqaure distance between two set of points
    #pts [n,3]
    #pts2 [m,3]

    device = pts_co.device

    n = pts_co.shape[0]
    n2 = pts_co2.shape[0]

    dist = torch.cdist(pts_co, pts_co2, p=2.0) #.to(device)
    
    return dist #[n,m]


def farest_point_sampling(pts_co, nb_sample, distance = None):
    #To compute FPS and sample indices
    #pts_co [b,n,3]

    device = pts_co.device

    n = pts_co.shape[1]
    b = pts_co.shape[0]

    samples = torch.empty([b, nb_sample], dtype=torch.long).to(device)
    dist = torch.zeros([b,n,nb_sample]).to(device)
    
    idx = torch.randint(n,[b])
    batch = torch.arange(b,dtype=torch.long)

    if distance is None:
        d0 = compute_distance(pts_co, pts_co[batch, idx, :].view(b,1,3))
    else:
        d0 = distance[batch, idx, :]
    d0 = d0.view(b,n)

    for i in range(nb_sample):
        idx = torch.argmax(d0,dim=1)
        samples[:,i] = idx
        #new_pts_co[:,i] = pts_co[batch,idx,:]

        if distance is None:
            d1 = compute_distance(pts_co, pts_co[batch, idx, :].view(b,1,3)).squeeze()
        else:
            d1 = distance[batch, idx, :]
        d1 = d1.view(b,n)
        dist[:,:,i] = d1
        d0 = torch.min(d0, d1)

    return samples, dist #[b,nb_sample], [b,nb_sample,3] [b,n,nb_sample]


def ball_query(radius, dist, group_size):
    #To group points within query balls around sampled points
    #dist [b,n,m]

    device = dist.device

    b = dist.shape[0]
    n = dist.shape[1]
    m = dist.shape[2]

    group = torch.arange(n, dtype=torch.long).repeat([b,m,1]).to(device)
    group[dist.permute(0,2,1) > radius] = n
    group, indices = torch.sort(group, dim=2)
    group = group[:,:,0:group_size]
    replace = group[:,:,0].view(b,m,1).repeat([1,1,group_size]).to(device)
    outside = group == n
    group[outside] = replace[outside]

    return group # [b, m, group_size]

def pts_index_query(pts,index):
    #pts [b,n,c]
    #index [b,l1,..,ln]

    device = pts.device

    b = pts.shape[0]

    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)

    new_pts = pts[batch_indices, index, :]

    return new_pts # [b,l1,..,ln,c]

def grouping(new_pts_co, samples, pts_fea, pts_co, radius, dist, group_size):
    #To make new point features by adding relative coordinates to sampled points
    #pts_co [b,n,3]
    #new_pts_co [b,m,3]
    #pts_fea [b,n,c]
    #dist [b,n,m]

    device = new_pts_co.device

    b = pts_co.shape[0]
    m = new_pts_co.shape[1]
    c = pts_fea.shape[2]

    group = ball_query(radius, dist, group_size)
    new_pts_fea = pts_index_query(pts_fea,samples).to(device).view(b,m,1,c).repeat([1,1,group_size,1])
    relative_co = (pts_index_query(pts_co,group) - new_pts_co.view(b,m,1,3).repeat([1,1,group_size,1])).to(device)
    new_pts_fea = torch.cat((new_pts_fea, relative_co), 3)

    return new_pts_fea #[b,m,g,c+3]

def sampling_grouping(pts_co, pts_fea, nb_sample, radius, group_size, distance = None, sample = None):
    #to call the samplig and grouping function and gather results 
    #pts_co [b,n,3]
    #pts_fea [b,n,c]
    #distance [b,N,n] or None
    #sample [b,n] or None
    
    b = pts_co.shape[0]
    
    if radius is None:
        new_pts_fea = grouping_all(pts_co, pts_fea)
        return None, new_pts_fea, None, None

    if distance is not None:
        distance = distance_reshape(distance, sample)

    samples, dist = farest_point_sampling(pts_co, nb_sample, distance)
    new_pts_co = pts_index_query(pts_co, samples)
    new_pts_fea = grouping(new_pts_co, samples, pts_fea, pts_co, radius, dist, group_size)

    return new_pts_co, new_pts_fea, dist, samples #[b,m,3], [b,m,g,c+3], [b,n,m], #[b,m]

def grouping_all(pts_co, pts_fea):
    # to group all points into the same group
    #pts_co [b,n,3]
    #pts_fea [b,n,c]
    
    device = pts_co.device
    
    new_pts_fea = torch.unsqueeze(pts_fea.clone().to(device), 1)
    
    return new_pts_fea #[b,1,n,c+3]

def distance_reshape(dist, samples):
    #dist [b,n,m]
    #samples [b,m]

    device = dist.device

    new_dist = pts_index_query(dist, samples)

    return new_dist #[b,m,m]