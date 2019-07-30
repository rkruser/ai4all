#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms, utils, models
import torch.nn.functional as F
#from data_loader import LeafSnapLoader
#import torch.optim as optim
#import argparse
import torch.nn as nn
#import torch.nn.init as init
#import os

from train_test import train, test, visualize_conv_layers
from test_on_new import test_on_new
import argparse

'''
For the Leafsnap project:
    - Clone the ai4all github project into an ai4all folder on your computer
    - Go to http://leafsnap.com/dataset/ and download the leafsnap dataset as a tar file
    - Use 7-Zip or another app to decompress the tar file. Place the leafsnap dataset in a subfolder
        of the ai4all project called "leafsnap-dataset"
    - Create a subfolder called "models" if one does not already exist
    - Create and train your own neural network
        by editing this file and running it.
    - Place your own leaf pictures in a subfolder and use test_network to classify them.
        The folder structure must look like
        ai4all
          [code files]
          models
          leafsnap-dataset
          pics
             my_pics  (A nested folder under pics is necessary for formatting reasons)
                pic1.jpg
                pic2.png
                etc.
    - You will need a gpu machine to train on the full dataset. The lab computers have gpus.
'''



# Fill in this class with a network you've designed
# You can change the class name as desired, and create multiple classes with different network structures
# See https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py for an example of a network
#  See https://pytorch.org/docs/stable/nn.html for documentation on network layers
class Your_Network(nn.Module):
    def __init__(self):
        super(Your_Network, self).__init__() # Initialize superclass
        # Your network layers here
        self.c1 = nn.Conv2d(3,64,5,stride = 1)
        
        self.c2 = nn.Conv2d(64,128,5,stride = 2)
        
        self.c3 = nn.Conv2d(128,256,3,stride = 1)
        
        self.c4 = nn.Conv2d(256,256,3,stride = 2)
        
        self.c5 = nn.Conv2d(256,256,3, stride = 2)
        
        self.c6 = nn.Conv2d(256,256,3, stride = 2)
        
        self.c7 = nn.Conv2d(256,256,3, stride = 2)
        
        self.c8 = nn.Linear(6400,512)
        
        self.c9 = nn.Linear(512,185)
        
    def forward(self, x):
        # Erase the word "pass" and place your code here
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.c7(x)
        
        # batchsize x 256 x 5 x 5
        x = F.relu(x)
        x = x.view(x.size(0), -1) # reshapes to batchsize x 6400
        
        x = self.c8(x)
        x = F.relu(x)
        x = self.c9(x)
        return x
        # The last output layer should have a tensor shape of batch_size x 185
        #  This should be the result of an nn.Linear layer
        #  Do not use a nonlinearity on the last layer



def train_network(use_alexnet = False, gpuid=0):
    # Adjust the training options as necessary
    if use_alexnet:
        print("Using alexnet")
    else:
        print("Using custom network")
        
    training_options = {
        'learning_rate': 0.0002,
        'training_epochs': 100,
        'batch_size': 64,
        'adam_beta_1': 0.5,
        'adam_beta_2': 0.999,
        'save_path': './models/alex.pth' if use_alexnet else './models/your_network.pth',        
        'continue_training_from': None, #Path to a saved neural net model
        'image_transform': transforms.Compose([
                    transforms.Resize((224,224)), #Make sure size matches model input size
                    # transforms.Pad(0),
                    # transforms.CenterCrop(224)
                    # See https://pytorch.org/docs/stable/torchvision/transforms.html
                    #   for more transforms
                ]),
        'data_source': ['lab', 'field'], # Whether to train on data from the lab or the field, or both
        'device': 'cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu',
        'printEvery': 10, #How often to print batch summaries
        'workers': 0
    }
    print(training_options)
    
    if use_alexnet:
        network = models.alexnet(pretrained=False, num_classes=185)
    else:
        network = Your_Network()
    train(network, training_options)
    

def test_network_on_testset(use_alexnet=False, gpuid=0):
    network = Your_Network()
    #network = models.alexnet(pretrained=False, num_classes=185)
    testing_options = {
        'batch_size': 16,       
        'load_from': './models/your_network_epochs_100.pth', #Path to a saved neural net model
        'image_transform': transforms.Compose([
                    transforms.Resize((224,224)), #Make sure size matches model input size
                    # transforms.Pad(0),
                    # transforms.CenterCrop(224)
                    # See https://pytorch.org/docs/stable/torchvision/transforms.html
                    #   for more transforms
                ]),
        'data_source': ['field'], # Whether to train on data from the lab or the field, or both
        'device': 'cuda:'+str(gpuid) if torch.cuda.is_available() else 'cpu',
        'workers': 0,
        'show_batches': [0]
    }
    test(network, testing_options)


def test_network_on_new():
    network = Your_Network()
    #network = models.alexnet(pretrained=False, num_classes=185)
    network.load_state_dict(torch.load('./models/your_network_epochs_100.pth'))
    test_on_new_options = {
        'folder': './pics',
        'image': None
    }
    test_on_new(network, test_on_new_options)
    
def visualize_network():
    network = Your_Network()
    #network = models.alexnet(pretrained=False, num_classes=185)
    #network.load_state_dict(torch.load('./models/your_network_epochs_100.pth')) 
    visualize_conv_layers(network.c4, 'c4_random.png')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_alexnet', action='store_true')
    parser.add_argument('--gpuid', type=int, default=0)
    opt = parser.parse_args()
    print(opt)
    
    #train_network(opt.use_alexnet, opt.gpuid)
    #test_network_on_new()
    #test_network_on_testset()
    #visualize_network()








