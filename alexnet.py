import torch
from torchvision import transforms, utils, models
from data_loader import LeafSnapLoader
import torch.optim as optim
import argparse
import torch.nn as nn
import torch.nn.init as init

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
		init.normal(m.weight, std=1e-2)
		# init.orthogonal(m.weight)
		#init.xavier_uniform(m.weight, gain=1.4)
		if m.bias is not None:
			init.constant(m.bias, 0.0)


def train(opt):
	trans = transforms.Resize((224,224)) #Needs to be 224
	dataset =  LeafSnapLoader(transform=trans)
	loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)	

	network = models.alexnet(pretrained=False, num_classes=185)
	if opt.loadfrom != '':
		network.load_state_dict(torch.load(opt.loadfrom))
	else:
		network.apply(weights_init)
	network = network.to(opt.device)

	optimizer = optim.Adam(network.parameters(), lr=opt.lr, betas=(opt.beta1,opt.beta2))
	lossfunc = nn.CrossEntropyLoss()


	accuracyMeter = AverageMeter()
	lossMeter = AverageMeter()

	for epoch in range(opt.nepochs):
		accuracyMeter.reset()
		lossMeter.reset()
		for i, data in enumerate(loader):
			# Extract training batches
			labels = data['species_index']
			ims = data['image']
			device_labels = labels.to(opt.device)
			device_ims = ims.to(opt.device)

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

			if i%100 == 0:
				accuracy = correct/batch_size
				print("Epoch {0}, batch {1}, batch_loss={2}, batch_accuracy={3}".format(
					epoch,i,loss.item(),accuracy))
		print("==================================")
		print("Epoch {0}, average batch loss: {1}, average batch accuracy: {2}".format(epoch, 
			 lossMeter.average(), accuracyMeter.average()))
		print("==================================")

	torch.save(network.state_dict(), opt.outpath)






if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--outpath', default='./models/alexnet.pth', help='Where to save the trained model')
	parser.add_argument('--nepochs', type=int, default=5, help='Number of training epochs')
	parser.add_argument('--device', default='cpu', help='Device to train on: cpu | cuda:0 | cuda:1 | ... | cuda:N')
	parser.add_argument('--batchsize', type=int, default=32, help='batch size')
	parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
	# 0.0002 works much better than the Adam default lr of 0.001
	parser.add_argument('--beta1', type=float, default=0.5, help='Adam parameter beta1')
	parser.add_argument('--beta2', type=float, default=0.999, help='Adam parameter beta2')
	parser.add_argument('--loadfrom', default='', help='Path to pre-existing model')
	opt = parser.parse_args()

	train(opt)

