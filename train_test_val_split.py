# Split the leafsnap-dataset-images.txt into training, testing, and validation sets

import os
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from data_loader import ClassLoader

filepath = './leafsnap-dataset/leafsnap-dataset-images.txt'

register_df = pd.read_csv(filepath, sep='\t')
targets = register_df['species']
npts = len(targets)
indices = np.arange(npts)

train_p = 0.8
val_p = 0.1
test_p = 1-(train_p + val_p)

# Stratify into train/validation/test
train_inds, valtest_inds, _, valtest_targets = train_test_split(indices, targets,
	stratify=targets, train_size=train_p)
validation_inds, test_inds = train_test_split(valtest_inds, stratify=valtest_targets, test_size=test_p/(val_p+test_p))

train_select = np.zeros(npts, dtype='int')
train_select[train_inds] = 1
val_select = np.zeros(npts, dtype='int')
val_select[validation_inds] = 1
test_select = np.zeros(npts, dtype='int')
test_select[test_inds] = 1

# Create new columns
register_df.insert(5,'species_index',0)
register_df.insert(6,'train',train_select)
register_df.insert(7,'val',val_select)
register_df.insert(8,'test',test_select)

# Reformat the species names
register_df['species'] = register_df['species'].apply(lambda s: s.replace(' ', '_').lower())

# Label each point with the species index
classes = ClassLoader()
register_df['species_index'] = register_df['species'].apply(classes.str2ind)

# Save to file
register_df.to_csv('./leafsnap-dataset-images-augmented.txt', sep='\t', index=False)


print(register_df[['species','train','val','test']].groupby('species').sum())