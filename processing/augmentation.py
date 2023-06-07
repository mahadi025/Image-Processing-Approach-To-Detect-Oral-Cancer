import os
import Augmentor
import splitfolders
import numpy as np
# %%
data_dir='../dataset/ideal_high_pass_images'
# %%
diseases_name = []
for image_class in os.listdir(data_dir):
    diseases_name.append(image_class)
print(diseases_name)
print(f'Total Disease: {len(diseases_name)}')
# %%
total_images = 0
for image_class in os.listdir(data_dir):
    print(image_class+':'+str(len(os.listdir(data_dir+'/'+image_class))))
    total_images += len(os.listdir(data_dir+'/'+image_class))
print(f'Total Images:{total_images}')
# %%
for disease in diseases_name:
    p = Augmentor.Pipeline(
        source_directory=data_dir+'/'+disease,
        output_directory='../../augmented_images/'+disease)

    p.flip_left_right(probability=1)

    p.flip_top_bottom(probability=1)

    p.rotate_random_90(probability=1)

    p.rotate_random_90(probability=1)

    # p.random_brightness(probability=1, min_factor=.5, max_factor=1.75)

    p.sample(len(os.listdir(data_dir+'/'+disease))*5)