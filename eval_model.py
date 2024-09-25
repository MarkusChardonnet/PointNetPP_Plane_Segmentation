from dataset import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pointnet_part import PointNet2
import sys

def eval_model(split, model, element):

    root = os.getcwd()
    dataset_name='shapenetpart'
    class_ = 'airplane'

    d = Dataset(root=root, dataset_name=dataset_name, class_choice=class_, 
                        num_points=2048, split=split, random_rotate=False, load_name=True)

    ps, lb, sg, n, f = d[element]

    print('Evaluating model on element ', element, ' of ', split, ' datatset.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    pts = torch.unsqueeze(ps,dim=0).to(device)
    output = model(pts)
    print('Evaluation done.')
    predicted = (torch.argmax(output, dim=2)).type(torch.LongTensor)
    predicted = torch.squeeze(predicted)

    print(sg)
    print(output)

    well_classified = torch.sum((sg == predicted).type(torch.LongTensor))
    print(well_classified, ' points out of 2048.')

    return well_classified

def main():
    split = sys.argv[1]
    element = int(sys.argv[2])
    PATH = sys.argv[3]
    model = PointNet2(4)
    chp = torch.load(PATH)
    model.load_state_dict(chp['model_state_dict'])
    eval_model(split, model, element)

if __name__ == '__main__':
    main()