import torch
from torchvision import transforms, utils, models
from data_loader import LeafSnapLoader
import torch.optim as optim
import argparse
import torch.nn as nn
import torch.nn.init as init


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


def train(outpath, nepochs, device):
	trans = transforms.Resize((224,224))
	data =  LeafSnapLoader(transform=trans)
	loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=4)	

	network = models.alexnet(pretrained=False, num_classes=185)
	network.apply(weights_init)
	network = network.to(device)

	optimizer = optim.Adam(network.parameters(), lr=0.002, betas=(0.5,0.999))
	lossfunc = nn.CrossEntropyLoss()

	for epoch in range(nepochs):
		print("Epoch",epoch)
		for i, data in enumerate(loader):
			labels = data['species_index']
			ims = data['image']
			device_labels = labels.to(device)
			device_ims = ims.to(device)

			network.zero_grad()
			predictions = network(device_ims)
			loss = lossfunc(predictions, device_labels)
			loss.backward()
			optimizer.step()

			if i%10 == 0:
				_, maxinds = predictions.to('cpu').max(1)
				correct = torch.sum(maxinds == labels).item()
				total = predictions.size(0)
				accuracy = correct/total

				print("Epoch {0}, batch {1}, batch_loss={2}, batch_accuracy={3}".format(
					epoch,i,loss.item(),accuracy))

	torch.save(network.state_dict(), outpath)






if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--outpath', required=True, help='Where to save the trained model')
	parser.add_argument('--nepochs', type=int, default=5, help='Number of training epochs')
	parser.add_argument('--device', default='cpu', help='Device to train on: cpu | cuda:0 | cuda:1 | ... | cuda:N')
	opt = parser.parse_args()

	train(opt.outpath, opt.nepochs, opt.device)

