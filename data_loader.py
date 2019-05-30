import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class LeafSnapLoader(Dataset):
	def __init__(self,csv_file='./leafsnap-dataset/leafsnap-dataset-images.txt',leafsnap_root='./leafsnap-dataset', source=['lab','field'],transform=None):
		self.leafsnap_root = leafsnap_root
		self.transform = transform
		self.frames = pd.read_csv(csv_file,sep='\t')
		self.frames = self.frames.loc[self.frames['source'].isin(source)]
		
		self.pil2tensor = transforms.ToTensor()
		
	def __len__(self):
		return len(self.frames)

	def __getitem__(self,idx):
		file_id = self.frames.iloc[idx,0]
		image_path = self.frames.iloc[idx,1]
		segmented_path = self.frames.iloc[idx,2]
		species = self.frames.iloc[idx,3]
		source = self.frames.iloc[idx,4]
		
		
		image = Image.open(self.leafsnap_root + '/' + image_path)
		segmented = Image.open(self.leafsnap_root + '/' + segmented_path)

		if self.transform:
			image = self.transform(image)
			segmented = self.transform(segmented)

		image = self.pil2tensor(image)
		segmented = self.pil2tensor(segmented)
		
		sample = {'file_id': file_id, 'species': species,
		'source': source, 'image': image, 'segmented': segmented}
		
		return sample


#test
if __name__ == '__main__':
	trans = transforms.Resize((256,256)) #use tranforms.Compose to add cropping etc.
	data =  LeafSnapLoader(transform=trans)
	loader = torch.utils.data.DataLoader(data, batch_size=5, shuffle=True, num_workers=4)
	iterator=loader.__iter__()
	sample = iterator.next()
	print(sample['file_id'])
	print(sample['species'])
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
	
