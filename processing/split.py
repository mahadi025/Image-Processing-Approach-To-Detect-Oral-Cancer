import os
import Augmentor
import splitfolders
import numpy as np
# %%
loc = '../dataset/augmented_images/'

# os.makedirs('../dataset/split/train')
# os.makedirs('../dataset/split/val')
# os.makedirs('../dataset/split/test')

os.makedirs('../dataset/new_split/train')
os.makedirs('../dataset/new_split/test')

splitfolders.ratio(loc, output='../dataset/new_split',
                   ratio=(0.80, .20))