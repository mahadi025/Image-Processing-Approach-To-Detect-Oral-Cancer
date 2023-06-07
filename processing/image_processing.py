import os
import cv2
import numpy as np
from PIL import Image
# %%
def resize(img):
    img=cv2.resize(img,(224,224))
    return img

def contrast_adjustment(img, alpha=1.2, beta=1):
    # alpha = 1.5  # Contrast control
    # beta = 10  # Brightness control
    output = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return output

def ideal_high_pass_filter(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    cutoff_frequency = 100
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img-img_back
# %%
# Renamed
original_dir = '../dataset/original'
renamed_images_dir='../dataset/renamed_images'
if not os.path.isdir(renamed_images_dir):
    os.makedirs(f'{renamed_images_dir}/cancer')
    os.makedirs(f'{renamed_images_dir}/non-cancer')
for image_class in os.listdir(original_dir):
    k=1
    l=1
    for image in os.listdir(os.path.join(original_dir, image_class)):
        image_path = os.path.join(original_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            if image_class == 'cancer':
                cv2.imwrite(os.path.join(renamed_images_dir+'/cancer','oral_cancer_' +str(k)+'.jpg'), img)
                k+=1
            else:
                cv2.imwrite(os.path.join(renamed_images_dir+'/non-cancer', 'non_cancer_'+str(l)+'.jpg'), img)
                l+=1
        except Exception as e:
            print(e)

# %%
# Resized
renamed_images_dir='../dataset/renamed_images'
resized_images_path='../dataset/resized_images'
if not os.path.isdir(resized_images_path):
    os.makedirs(f'{resized_images_path}/cancer')
    os.makedirs(f'{resized_images_path}/non-cancer')
for image_class in os.listdir(resized_images_path):
    for image in os.listdir(os.path.join(renamed_images_dir, image_class)):
        image_path = os.path.join(renamed_images_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            img=resize(img)
            if image_class == 'cancer':
                cv2.imwrite(os.path.join(resized_images_path+r'\\cancer', image), img)
            else:
                cv2.imwrite(os.path.join(resized_images_path+r'\\non-cancer', image), img)
        except Exception as e:
            print(e)
# %%
# Contrast Adjustment
resized_images_path='../dataset/resized_images'
contrast_adjustment_path='../dataset/contrast_adjustment_images'
if not os.path.isdir(contrast_adjustment_path):
    os.makedirs(f'{contrast_adjustment_path}/cancer')
    os.makedirs(f'{contrast_adjustment_path}/non-cancer')
for image_class in os.listdir(resized_images_path):
    for image in os.listdir(os.path.join(resized_images_path, image_class)):
        image_path = os.path.join(resized_images_path, image_class, image)
        try:
            img = cv2.imread(image_path)
            img=contrast_adjustment(img, alpha=1.3, beta=1)
            if image_class == 'cancer':
                cv2.imwrite(os.path.join(contrast_adjustment_path+r'\\cancer', image), img)
            else:
                cv2.imwrite(os.path.join(contrast_adjustment_path+r'\\non-cancer', image), img)
        except Exception as e:
            print(e)
# %%
# GrayScale
contrast_adjustment_path='../dataset/contrast_adjustment_images'
grayscale_path='../dataset/grayscale_images'
if not os.path.isdir(grayscale_path):
    os.makedirs(f'{grayscale_path}/cancer')
    os.makedirs(f'{grayscale_path}/non-cancer')
for image_class in os.listdir(contrast_adjustment_path):
    for image in os.listdir(os.path.join(contrast_adjustment_path, image_class)):
        image_path = os.path.join(contrast_adjustment_path, image_class, image)
        try:
            img = cv2.imread(image_path,0)
            if image_class == 'cancer':
                cv2.imwrite(os.path.join(grayscale_path+r'\\cancer', image), img)
            else:
                cv2.imwrite(os.path.join(grayscale_path+r'\\non-cancer', image), img)
        except Exception as e:
            print(e)
# %%
# Ideal High Pass
grayscale_path='../dataset/grayscale_images'
ideal_high_pass_images_path='../dataset/ideal_high_pass_images'
if not os.path.isdir(ideal_high_pass_images_path):
    os.makedirs(f'{ideal_high_pass_images_path}/cancer')
    os.makedirs(f'{ideal_high_pass_images_path}/non-cancer')
for image_class in os.listdir(grayscale_path):
    for image in os.listdir(os.path.join(grayscale_path, image_class)):
        image_path = os.path.join(grayscale_path, image_class, image)
        try:
            img = cv2.imread(image_path,0)
            img=ideal_high_pass_filter(img)
            if image_class == 'cancer':
                cv2.imwrite(os.path.join(ideal_high_pass_images_path+r'\\cancer', image), img)
            else:
                cv2.imwrite(os.path.join(ideal_high_pass_images_path+r'\\non-cancer', image), img)
        except Exception as e:
            print(e)