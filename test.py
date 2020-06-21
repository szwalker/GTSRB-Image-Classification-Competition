'''
NYU Computer Vision CS 480

** Description **
This is the testing code.
This code should be used to run on the testing dataset.

** Usage **
Please place this file in the same directory as Net.py.
Please modify the following variables before running the code:
    PATH_TO_TEST_GTSRB (the path to the GTRSB folder in the test set)
    PATH_TO_TEST_CSV_LABEL (the path to the .csv file with test set labels)
    PATH_TO_NET (the path to the net.pth file)

** Outputs **
This file will produce the final testing accuracy in the standard output.
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
class GTSRB_Test(data.Dataset):
    def __init__(self,test_folder_path,test_label_path,transform=None):
        self.test_folder_path = test_folder_path
        self.test_label_path = test_label_path
        self.transform = transform
        print(glob(os.path.join("{}/Final_Test/Images".format(self.test_folder_path), "*.csv")))
        roi_csv = open(glob(os.path.join("{}/Final_Test/Images".format(self.test_folder_path), "*.csv"))[0],'r')
        roi_csv.readline()
        lines = roi_csv.readlines()
        roi_csv.close()
        d = {}
        for _ in lines:
            info = _.rstrip().split(";")
            d[info[0]] = (int(info[3]), int(info[4]), int(info[5]), int(info[6]))
        img_paths = sorted(glob(os.path.join("{}/Final_Test/Images".format(self.test_folder_path), "*.ppm")))
        self.data = [(_,d[_.split("/")[-1]]) for _ in img_paths]
        del d,lines
        label_d = {}
        label_csv_file = open(self.test_label_path,'r')
        label_csv_file.readline()
        lines = label_csv_file.readlines()
        label_csv_file.close()
        for _ in lines:
            info = _.rstrip().split(";")
            label_d[info[0]]=int(info[-1])
        self.label = [label_d[_.split("/")[-1]] for _ in img_paths]
        del label_d,lines

    def __getitem__(self, i):
        (img_path,roi),label = self.data[i],self.label[i]
        im = Image.open(img_path).crop(roi)
        if self.transform: im = self.transform(im)
        return im, label

    def __len__(self): return len(self.data)

    def get_orig_img_path(self,i): return self.data[i]

def get_transform():
    return T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

if __name__ == "__main__":
    # File Paths
    PATH_TO_TEST_GTSRB = '../GTSRB_Final_Test_Images/GTSRB'
    PATH_TO_TEST_CSV_LABEL = './GT-final_test.csv'
    PATH_TO_NET = './net.pth'
    # use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initlize test dataset
    DS_test = GTSRB_Test(PATH_TO_TEST_GTSRB, PATH_TO_TEST_CSV_LABEL,transform=get_transform())

    DL_test = data.DataLoader(DS_test, batch_size=50, shuffle=False, num_workers=12)
    print("Test set length:",len(DS_test))
    # initialize net & load state dict
    net = Net().to(device)
    net_state_dict = torch.load(PATH_TO_NET)
    net.load_state_dict(net_state_dict,strict=False)

    # parallel computing setup for multiple GPU
    GPU_count = torch.cuda.device_count()
    print("Currently using: {:2d} GPU".format(GPU_count))
    if GPU_count > 1:
        # increase parallelism
        net = nn.DataParallel(net)
        print("\t|- Increased parallelism enabled!")

    # initialize optimizer and loss function
    loss_function = nn.CrossEntropyLoss()

    # performance testing
    net.eval()
    with torch.no_grad():
        correct = 0
        total_val_loss = 0
        # extract the predicted label and compare it to ground truth
        for batch_ind,(X,y) in enumerate(DL_test):
            X = X.to(device)
            y = y.to(device)
            t = time.time()
            outputs = net(X) # (tensor, grad_fn)
            loss = loss_function(outputs,y)
            total_val_loss += loss.item()
            correct += sum([1 for i in range(len(y)) if torch.argmax(outputs[i]) == y[i]])
            print("Evaluation Progress: {:3d}/{:3d}".format(batch_ind,len(DL_test)))
        accuracy = correct / len(DS_test)
        loss = total_val_loss / len(DS_test)
        print("[Model Performance on Test Set]\n\t|- Accuracy: {:5f}\n\t|- Loss: {:5f}".format(accuracy,loss))
