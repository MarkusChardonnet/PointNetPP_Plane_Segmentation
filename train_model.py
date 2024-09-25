from dataset import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pointnet_part import PointNet2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
import sys
import eval_model as ev


def train_model(TRAINING_SIZE, EPOCHS, START, PATH):
    
    start_time = time.time()

    dataset_name='shapenetpart'
    class_ = 'airplane'
    split = 'train'
    labels = 8

    root = os.getcwd()
    save_root = os.path.join("model")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Loading training data..')
    d = Dataset(root=root, dataset_name=dataset_name, class_choice=class_, 
                            num_points=2048, split=split, random_rotate=False, load_name=True)

    print("datasize:", d.__len__())

    pts = torch.empty([TRAINING_SIZE, 2048, 3])
    seg = torch.empty([TRAINING_SIZE, 2048])
    for i in range(TRAINING_SIZE):
        ps, lb, sg, n, f = d[i + START]
        pts[i] = ps
        seg[i] = sg
    seg = seg.type(torch.LongTensor)
    print('Done!')

    print('Checking for cuda..')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('Cuda is available. Loading data on GPU...')
        pts = pts.to(device)
        seg = seg.to(device)
        print('Done!')
    else: 
        print('Cuda is not available')

    print('Creating model and optimizer instance..')
    model = PointNet2(4).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    print('Done!')
    

    if PATH is not None:
        print('Loading existing model...')
        chp = torch.load(PATH)
        model.load_state_dict(chp['model_state_dict'])
        optimizer.load_state_dict(chp['optimizer_state_dict'])
        print('Done!')
    else:
        print('No existing model, starting from scratch.')

    print('Preparing for training')
    good_model = 900
    BATCH_SIZE = 4
    model.train()
    dataset = TensorDataset(pts, seg)
    loader = DataLoader(dataset, batch_size = BATCH_SIZE)

    print('Starting training')
    for i in range(EPOCHS):
        print('Current epoch : ', EPOCHS)
        for batch_ndx, sample in enumerate(loader):
            print('Batch ', batch_ndx, 'being processed..')
            inputs, labels = sample
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(0,2,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_ndx % 8 == 0 and batch_ndx != 0:
                m = ev.eval_model('test', model, 0)
                if m > good_model:
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_root + '/epoch{}batch{}.pt'.format(i, batch_ndx))
                    good_model = m
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_root + '/epoch{}.pt'.format(i))
    
    
    print(time.time() - start_time)
    
def main():
    TRAINING_SIZE = int(sys.argv[1])
    EPOCHS = int(sys.argv[2])
    START = int(sys.argv[3])
    PATH = None
    if len(sys.argv) > 4:
        PATH = sys.argv[4]
    train_model(TRAINING_SIZE, EPOCHS, START, PATH)

if __name__ == '__main__':
    main()