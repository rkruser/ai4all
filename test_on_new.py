# Test on new pics

import torch
from torchvision import models
import argparse
import torchvision.transforms.functional as func
from PIL import Image
from data_loader import ClassLoader

parser = argparse.ArgumentParser()
parser.add_argument('--loadfrom', default='', help='The model to load')
parser.add_argument('--image', default='', help='The image to classify')
parser.add_argument('--folder', default='', help='A whole folder of images to classify')
opt = parser.parse_args()

classes = ClassLoader()

network = models.alexnet(pretrained=False, num_classes=185)
network.load_state_dict(torch.load(opt.loadfrom))	
network.eval()

image = Image.open(opt.image)
image = func.resize(image, (224,224))
image.show()
image = func.to_tensor(image).unsqueeze(0)

result = network(image).squeeze()

_, arg = torch.max(result,0)
arg = arg.item()

print("Index: {0}, Classification: {1}".format(arg,classes.ind2str(arg).replace('_',' ')))





