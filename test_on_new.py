# Test on new pics

import torch
from torchvision import models, datasets, transforms
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


class CustomImageFolder(datasets.ImageFolder):
	def __init__(self, location, transform=None):
		super(CustomImageFolder,self).__init__(location, transform=transform)

	def __getitem__(self, index):
		x, y = super(CustomImageFolder, self).__getitem__(index)
		return {'image':x, 'class':y, 'file':self.imgs[index][0]}


if opt.folder != '':
	dset = CustomImageFolder(opt.folder, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
	loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=1)	
#	iterator=loader.__iter__()
#	sample = iterator.next()	
#	print(sample)
	for i, data in enumerate(loader):
		image = data['image']
		file = data['file'][0]
#		func.to_pil_image(image.squeeze()).show(title=file) #visualize the image #title doesn't work
		result = network(image).squeeze()
		_, arg = torch.max(result,0)
		arg = arg.item()
		print("File: {0}, Index: {1}, Classification: {2}".format(file, arg, classes.ind2str(arg).replace('_',' ')))

elif opt.image != '':
	image = Image.open(opt.image)
	image = func.resize(image, (224,224))
	image.show()
	image = func.to_tensor(image).unsqueeze(0)

	result = network(image).squeeze()

	_, arg = torch.max(result,0)
	arg = arg.item()

	print("Index: {0}, Classification: {1}".format(arg,classes.ind2str(arg).replace('_',' ')))

else:
	print("Invalid image or folder")





