'''
NYU Computer Vision CS 480

** Description **
This is the training code.
Our approach is to split the given data set into two sets.
One set is the training set, contains 90% of randomly chose data from the data set.
The remaining 10% data is used for validation purpose.

** Usage **
Please modify the variable PATH_TO_GTSRB_FOLDER to the GTSTRB folder path in the training file before running the code.

** Outputs **
This file will produce 4 files to the current directory:
    * net.pth
    * training_loss_vs_epoch.png
    * validation_loss_vs_epoch.png
    * validation_accuracy_vs_epoch.png
'''

# libraries setup
import torch
import torchvision

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from glob import glob
from PIL import Image
from model import Net

# dataset
class GTSRB(data.Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_folders_paths = sorted(glob(os.path.join("{}/Final_Training/Images".format(self.path), "*")))
        for folders_paths in class_folders_paths:
            roi_csv = open(glob(os.path.join(folders_paths, "GT-*.csv"))[0],'r')
            roi_csv.readline()
            lines = roi_csv.readlines()
            d = {}
            for _ in lines:
                info = _.rstrip().split(";")
                d[info[0]] = (int(info[3]), int(info[4]), int(info[5]), int(info[6]))
            roi_csv.close()
            folder_img_paths = sorted(glob(os.path.join(folders_paths, "*.ppm")))
            self.data.extend([(_,int(folders_paths.split("/")[-1]), d[_.split("/")[-1]]) for _ in folder_img_paths])

    def __getitem__(self, i):
        img_path,label,roi = self.data[i]
        im = Image.open(img_path).crop(roi)
        if self.transform: im = self.transform(im)
        return im,label

    def __len__(self): return len(self.data)

    def get_orig_img_path(self,i): return self.data[i]

def get_transform():
    return T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

if __name__ == "__main__":
    # not allowing the data to be reproduced
    torch.manual_seed(0)

    # hyper-parameter
    EPOCHS = 23
    LEARNING_RATE = 1e-5
    ALPHA = 0.99
    EPS = 1e-08
    WEIGHT_DECAY = 0
    MOMEENTUM = 0

    PATH_TO_GTSRB_FOLDER = './GTSRB'

    # use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initlize dataset
    DS = GTSRB(PATH_TO_GTSRB_FOLDER, transform=get_transform())

    # split the initial training data into 90% training data and 10% validation data
    split_lengths = [int(len(DS) * 0.9), len(DS) - int(len(DS) * 0.9)]
    DS_train,DS_val = data.random_split(DS, split_lengths)

    print("[Data Spliting Result]\n\t|- Training Set: {:5d}\n\t|- Validation Set:{:5d}".format(len(DS_train),len(DS_val)))

    # create data loader for train and val set
    DL_train = data.DataLoader(DS_train, batch_size=50, shuffle=True, num_workers=12)
    DL_val = data.DataLoader(DS_val, batch_size=10, shuffle=True, num_workers=12)

    # initialize net
    net = Net().to(device)

    # parallel computing setup for multiple GPU
    GPU_count = torch.cuda.device_count()
    print("Currently using: {:2d} GPU".format(GPU_count))
    if GPU_count > 1:
        # increase parallelism
        net = nn.DataParallel(net)
        print("\t|- Increased parallelism enabled!")

    # initialize optimizer and loss function
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=ALPHA, eps=EPS, weight_decay=WEIGHT_DECAY, momentum=MOMEENTUM, centered=False)
    loss_function = nn.CrossEntropyLoss()

    # array for data ploting
    Train_Loss_axis = [None for _ in range(EPOCHS)]
    Val_Loss_axis = [None for _ in range(EPOCHS)]
    Val_Accuracy_axis = [None for _ in range(EPOCHS)]

    for epoch in range(EPOCHS):
        # train one epoch
        net.train()
        total_training_loss = 0
        for batch_ind, (X,y) in enumerate(DL_train):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            t = time.time()
            outputs = net(X)
            loss = loss_function(outputs,y)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()
            print('Epoch (training) {:2d} Batch: {:3d}/{:3d} Batch Loss: {:.5f} ({:.3f}s)'.format(
                epoch + 1,
                batch_ind + 1,
                len(DL_train),
                loss.item(),
                time.time() - t), end='\n')
        Train_Loss_axis[epoch] = total_training_loss / len(DS_train)
        print('Epoch [{:2d}/{:2d}] Training Loss: {:.5f}'.format(epoch + 1, EPOCHS, Train_Loss_axis[epoch]))

        # validation one epoch
        net.eval()
        with torch.no_grad():
            correct = 0
            total_val_loss = 0
            # extract the predicted label and compare it to ground truth
            for batch_ind,(X,y) in enumerate(DL_val):
                X = X.to(device)
                y = y.to(device)
                t = time.time()
                outputs = net(X) # (tensor, grad_fn)
                loss = loss_function(outputs,y)
                total_val_loss += loss.item()
                correct += sum([1 for i in range(len(y)) if torch.argmax(outputs[i]) == y[i]])
                print('Epoch (validation) {:2d} Batch: {:3d}/{:3d} Batch Loss: {:.5f} ({:.3f}s)'.format(
                    epoch + 1,
                    batch_ind + 1,
                    len(DL_val),
                    loss.item(),
                    time.time() - t), end='\n')
            Val_Accuracy_axis[epoch] = correct / len(DS_val)
            Val_Loss_axis[epoch] = total_val_loss / len(DS_val)
            print('Epoch [{:2d}/{:2d}] Validation Loss: {:.5f} Accuracy: {:.2f}'.format(epoch+1, EPOCHS, total_val_loss / len(DS_val), correct / len(DS_val)))

    # save model
    torch.save(net.state_dict(), './net.pth')
    print("Model saved!")

    # plot training loss, validation loss, and validation accuracy over epochs
    plt.style.use('seaborn-darkgrid')

    EPOCHS_axis = [_ for _ in range(1,EPOCHS+1)]
    plt.figure(0)
    plt.plot(EPOCHS_axis, Train_Loss_axis,label="Loss")
    plt.suptitle('Training Loss Over Epoch')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('training_loss_vs_epoch.png',dpi=1000)

    plt.figure(1)
    plt.plot(EPOCHS_axis, Val_Loss_axis,label="Loss")
    plt.suptitle('Validation Loss Over Epoch')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig('validation_loss_vs_epoch.png',dpi=1000)

    plt.figure(2)
    plt.plot(EPOCHS_axis, Val_Accuracy_axis,label="Accuracy")
    plt.suptitle('Validation Accuracy Over Epoch')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig('validation_accuracy_vs_epoch.png',dpi=1000)

    print("Images saved!")
