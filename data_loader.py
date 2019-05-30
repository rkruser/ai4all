import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Read a class file and transform between class label and string
class ClassLoader(object):
	def __init__(self, class_file='./leafsnap-dataset/classes.txt'):
		with open(class_file, 'r') as f:
			self.classes = f.read().splitlines()
#		for i,c in enumerate(self.classes):
#			self.classes[i] = c.replace('_',' ')

		self.string2index = {c:i for i,c in enumerate(self.classes)}
		self.index2string = {i:c for i,c in enumerate(self.classes)}

	def str2ind(self, s):
		return self.string2index[s]

	def ind2str(self, i):
		return self.index2string[i]




class LeafSnapLoader(Dataset):
	def __init__(self,csv_file='./leafsnap-dataset/leafsnap-dataset-images.txt',
		leafsnap_root='./leafsnap-dataset', 
		source=['lab','field'],
		class_file='./leafsnap-dataset/classes.txt',
		transform=None):
		self.leafsnap_root = leafsnap_root
		self.transform = transform
		self.frames = pd.read_csv(csv_file,sep='\t')
		self.frames = self.frames.loc[self.frames['source'].isin(source)]
		self.classes = ClassLoader(class_file)
		
		self.pil2tensor = transforms.ToTensor()
		
	def __len__(self):
		return len(self.frames)

	def __getitem__(self,idx):
		file_id = self.frames.iloc[idx,0]
		image_path = self.frames.iloc[idx,1]
		segmented_path = self.frames.iloc[idx,2]
		species = self.frames.iloc[idx,3]
		source = self.frames.iloc[idx,4]

		species_index = self.classes.str2ind(species.replace(' ', '_').lower())
		# The .lower() is necessary because pytorch feels the need to capitalize its classes automatically
		# It also replaces underscores with spaces automatically, hence the .replace()
		
		
		image = Image.open(self.leafsnap_root + '/' + image_path)
		segmented = Image.open(self.leafsnap_root + '/' + segmented_path)

		if self.transform:
			image = self.transform(image)
			segmented = self.transform(segmented)

		image = self.pil2tensor(image)
		segmented = self.pil2tensor(segmented)
		
		sample = {'file_id': file_id, 'species': species, 'species_index': species_index,
		'source': source, 'image': image, 'segmented': segmented}
		
		return sample


#test
if __name__ == '__main__':
	trans = transforms.Resize((256,256)) #use tranforms.Compose to add cropping etc.
	data =  LeafSnapLoader(transform=trans)
	loader = torch.utils.data.DataLoader(data, batch_size=200, shuffle=True, num_workers=4)
	iterator=loader.__iter__()
	sample = iterator.next()
	print(sample['file_id'])
	print(sample['species'])
	print(sample['species_index'])
	print(sample['source'])
	print(sample['image'].shape)
	print(sample['segmented'].shape)
	images = sample['image']
	segmented = sample['segmented']
	im = images[0,:,:,:]
	seg = segmented[0,:,:,:]
	
	toPIL = transforms.ToPILImage()
	im = toPIL(im)
	seg = toPIL(seg)
	
	im.show()
	seg.show()
	
