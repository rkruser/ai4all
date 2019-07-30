#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions called by project.py for training, testing, and visualizing
'''

import torch
#from torchvision import transforms, utils, models
from data_loader import LeafSnapLoader
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.utils as utils
#import argparse
import torch.nn as nn
import torch.nn.init as init
import os
import pickle
import PIL
#from PIL import Image

from data_loader import ClassLoader


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.value = 0

    def update(self, value, count=1):
        self.value += value
        self.count += count

    def average(self):
        return self.value / self.count

    def reset(self):
        self.count = 0
        self.value = 0


# I copy-pasted this from somewhere else. Not sure if this is a good init procedure -Ryen
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight, std=1e-2)
        # init.orthogonal(m.weight)
        #init.xavier_uniform(m.weight, gain=1.4)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def train(network, opt):
    image_transform = opt['image_transform']
    device = opt['device']
    nepochs = opt['training_epochs']
    data_source = opt['data_source']
    printEvery = opt['printEvery']
    workers = opt['workers']
    
    
    dataset_train =  LeafSnapLoader(mode='train', transform=image_transform, source=data_source)
    dataset_val = LeafSnapLoader(mode='val', transform=image_transform, source=data_source)
    dataset_test = LeafSnapLoader(mode='test', transform=image_transform, source=data_source)
    
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=opt['batch_size'], shuffle=True, num_workers=workers) 
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=opt['batch_size'], shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=opt['batch_size'], shuffle=True, num_workers=workers)

    # network = models.alexnet(pretrained=False, num_classes=185)
    if opt['continue_training_from'] is not None:
        network.load_state_dict(torch.load(opt['continue_training_from']))
    else:
        network.apply(weights_init)
    network = network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=opt['learning_rate'], betas=(opt['adam_beta_1'],opt['adam_beta_2']))
    lossfunc = nn.CrossEntropyLoss()

    # For saving later
    nameprefix, extension = os.path.splitext(opt['save_path'])

    accuracyMeter = AverageMeter()
    lossMeter = AverageMeter()
    valAccuracyMeter = AverageMeter()
    valLossMeter = AverageMeter()
    prevLoss = 1000000 # Just a large number
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    # Added network.train(), network.test()? 

    for epoch in range(nepochs):
        accuracyMeter.reset()
        lossMeter.reset()
        valAccuracyMeter.reset()
        valLossMeter.reset()
        network.train()
        for i, data in enumerate(trainloader):
            # Extract training batches
            labels = data['species_index']
            ims = data['image']
            device_labels = labels.to(device)
            device_ims = ims.to(device)

            # Train network
            network.zero_grad()
            predictions = network(device_ims)
            loss = lossfunc(predictions, device_labels)
            loss.backward()
            optimizer.step()

            # Compute statistics
            _, maxinds = predictions.max(1)
            maxinds = maxinds.to('cpu')
            correct = torch.sum(maxinds == labels).item()
            batch_size = predictions.size(0)
            accuracyMeter.update(correct, batch_size)
            lossMeter.update(loss.item())

            if i%printEvery == 0:
                accuracy = correct/batch_size
                print("batch {1} of {4}, batch_loss={2}, batch_accuracy={3}".format(
                    epoch,i,loss.item(),accuracy, len(trainloader)))

        # Validate on held-out data
        network.eval()
        for i, data in enumerate(valloader):
            labels = data['species_index']
            ims = data['image']
            device_labels = labels.to(device)
            device_ims = ims.to(device)

            predictions = network(device_ims)
            loss = lossfunc(predictions, device_labels)

            _, maxinds = predictions.max(1)
            maxinds = maxinds.to('cpu')
            correct = torch.sum(maxinds == labels).item()
            batch_size = predictions.size(0)
            valAccuracyMeter.update(correct, batch_size)
            valLossMeter.update(loss.item())


        print("==================================")
        print("Epoch {0}, train batch loss: {1}, train batch accuracy: {2}\n validation loss: {3}, validation accuracy: {4}".format(
            epoch, lossMeter.average(), accuracyMeter.average(), valLossMeter.average(), valAccuracyMeter.average()))
        print("==================================")

        train_accs.append(accuracyMeter.average())
        val_accs.append(valAccuracyMeter.average())
        train_losses.append(lossMeter.average())
        val_losses.append(valLossMeter.average())
        with open(nameprefix+'_savedvalues.pkl', 'wb') as fp:
            pickle.dump((train_accs, train_losses, val_accs, val_losses), fp)

        if valLossMeter.average() < prevLoss:
            prevLoss = valLossMeter.average()
            print("Saving best current model")
            torch.save(network.state_dict(), nameprefix+'_best_'+str(epoch)+extension)


    # Evaluate on held-out test
    testAccuracyMeter = AverageMeter()
    testLossMeter = AverageMeter()
    network.eval()
    for i, data in enumerate(testloader):
        labels = data['species_index']
        ims = data['image']
        device_labels = labels.to(device)
        device_ims = ims.to(device)

        predictions = network(device_ims)
        loss = lossfunc(predictions, device_labels)

        _, maxinds = predictions.max(1)
        maxinds = maxinds.to('cpu')
        correct = torch.sum(maxinds == labels).item()
        batch_size = predictions.size(0)
        testAccuracyMeter.update(correct, batch_size)
        testLossMeter.update(loss.item())
    print("Average test batch loss: {0}, Test accuracy: {1}".format(testLossMeter.average(), testAccuracyMeter.average()))

    torch.save(network.state_dict(), nameprefix+'_epochs_'+str(nepochs)+extension)


def test(network, opt):
    workers = opt['workers']
    image_transform = opt['image_transform']
    data_source = opt['data_source']
    device = opt['device']
    show_batches = opt['show_batches'] #list of batches to show
    
    dataset_test = LeafSnapLoader(mode='test', transform=image_transform, source=data_source)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=opt['batch_size'], shuffle=True, num_workers=workers)
    classes = ClassLoader()
    
    
    network.load_state_dict(torch.load(opt['load_from']))
    network = network.to(device)
    
    lossfunc = nn.CrossEntropyLoss()

    testAccuracyMeter = AverageMeter()
    testLossMeter = AverageMeter()
    network.eval()
    for i, data in enumerate(testloader):
        labels = data['species_index']
        ims = data['image']
        device_labels = labels.to(device)
        device_ims = ims.to(device)

        predictions = network(device_ims)
        loss = lossfunc(predictions, device_labels)

        _, maxinds = predictions.max(1)
        maxinds = maxinds.to('cpu')
        correct = torch.sum(maxinds == labels).item()
        batch_size = predictions.size(0)
        testAccuracyMeter.update(correct, batch_size)
        testLossMeter.update(loss.item())
        
        if i in show_batches:
            print_ims = utils.make_grid(ims, nrow=4)
            #print_image = F.to_pil_image(print_ims)
            utils.save_image(print_ims, './test_ims.png')
            #print_image.show()
            
            for im_idx in range(len(maxinds)):
                arg = maxinds[im_idx].item()
                print("Index: {0}, classification: {1}, true class: {2}, source: {3}".format( 
                      arg, 
                      classes.ind2str(arg),
                      classes.ind2str(labels[im_idx].item()),
                      data['source'][im_idx]
                      ))
                
            
        
    print("Average test batch loss: {0}, Test accuracy: {1}".format(testLossMeter.average(), testAccuracyMeter.average()))    


def visualize_conv_layers(conv_layer, file_name):
    weights = conv_layer.weight.data
    print(weights.size())
    #weights = weights.permute(1,0,2,3)
    if weights.size(1) > 3:
        weights = weights.mean(1).unsqueeze(1)
    
    print_ims = utils.make_grid(weights, nrow=16, normalize=True, scale_each=False)
    print_image = F.to_pil_image(print_ims)
    print_image = F.resize(print_image, 512, interpolation=PIL.Image.NEAREST)
    #utils.save_image(print_ims, file_name)
    print_image.save(file_name)
    print_image.show()


















