import numpy as np
import cv2
import os
import glob

import torch
import pretrain_utils
import torchvision_models

import matplotlib.pyplot as plt

"""
get imput img tensor from image folders
and visualize them
1. original event tensor (num_bins, H, W)
2. 1 + gaussian noise
3. 1 + gaussian noise + hot pixels
"""

# dst = r'../msj_img_folders/video100'
img_file_id = input("please input the index of image folders: ")
# id = int(img_file_id)
dst = os.path.join('/home/CBICR/msj/events-caption.pytorch/video2images/', 'video%s' % img_file_id)
n_frame_steps = 40
C, H, W = 3, 224, 224
model = torchvision_models.resnet152(pretrained='imagenet')
load_image_fn = pretrain_utils.LoadTransformImage(model)
# print(model)


# print(images)
# print(images.size())   # torch.Size([40, 3, 224, 224])

# print(model.input_size)   # [3, 224, 224]
# print(model.mean)   # [0.485, 0.456, 0.406]
# print(model.std)   # [0.229, 0.224, 0.225]
# print(type(model))   # <class 'torchvision.models.resnet.ResNet'>
# min, max = torch.min(images), torch.max(images)
# mean, std = torch.mean(images), torch.std(images)
# print("min = %f , max = %f" % (min, max))
# print("mean = %f , std = %f" % (mean, std))


num_bins = 5
# contrast threshold (CT)
CT_mean = 0.2
CT_std = 0.03
gauss_mean = 0.0
gauss_std = 0.1
hot_mean = 0.0
hot_std = 0.2
hot_p = 0.001
CT = np.random.randn(num_bins, H, W) * CT_std + CT_mean
CT[CT < 0.01] = 0.01
gauss_noise = np.random.randn(num_bins, H, W) * gauss_std + gauss_mean
hot_pixel = np.random.randn(num_bins, H, W) * hot_std + hot_mean
hot_mask = np.random.rand(num_bins, H, W)
hot_pixel[hot_mask > hot_p] = 0.0

image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
samples = np.round(np.linspace(
    0, len(image_list) - 1, (num_bins + 1) * n_frame_steps))
image_list = [image_list[int(sample)] for sample in samples]
# images = torch.zeros((len(image_list), C, H, W))
images = np.zeros((len(image_list), C, H, W))
for iImg in range(len(image_list)):
    img = load_image_fn(image_list[iImg])
    images[iImg] = img
# images = torch.mean(images, dim=1, keepdim=False)
#print(images.shape)   # (240, 3, 224, 224)
images = np.mean(images, axis=1, keepdims=False)
#print(images.shape)   # (240, 224, 224)

# img_diff = torch.zeros((len(image_list) // (num_bins+1), num_bins, H, W))
# event_tensor = torch.zeros((len(image_list) // (num_bins+1), num_bins, H, W))
# img_diff = torch.zeros((n_frame_steps, num_bins, H, W))
# event_tensor = torch.zeros((n_frame_steps, num_bins, H, W))
img_diff = np.zeros((n_frame_steps, num_bins, H, W))
event_tensor = np.zeros((n_frame_steps, num_bins, H, W))
for i in range(n_frame_steps):
    for j in range(num_bins):
        img_diff[i, j, :, :] = images[i * (num_bins + 1) + j + 1, :, :] - images[i * (num_bins + 1) + j, :, :]

event_tensor[img_diff > CT] = 1.0
event_tensor[img_diff < -CT] = -1.0
arr1 = event_tensor.copy()
#print(event_tensor)
#print(event_tensor.shape)   # (40, 5, 224, 224)
#print('\n')

event_tensor += gauss_noise
arr2 = event_tensor.copy()

event_tensor += hot_pixel
arr3 = event_tensor.copy()

#print(event_tensor)
#print(event_tensor.shape)   # (40, 5, 224, 224)

frame_idx = 0
bin_idx = 0
frame_idx = int(input("please input the frame index: "))
#e = arr1[frame_idx, bin_idx, :, :]
e = np.sum(arr1[frame_idx, :, :, :], axis=0)
#print(e.shape)   # (224, 224)
eps = 0.1
e_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
e_rgb[e < -eps] = (255, 0, 0)
e_rgb[e > eps] = (0, 0, 255)

#e_g = arr2[frame_idx, bin_idx, :, :]
e_g = np.sum(arr2[frame_idx, :, :, :], axis=0)
e_g_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
e_g_rgb[e_g < -eps] = (255, 0, 0)
e_g_rgb[e_g > eps] = (0, 0, 255)

#e_g_h = arr3[frame_idx, bin_idx, :, :]
e_g_h = np.sum(arr3[frame_idx, :, :, :], axis=0)
e_g_h_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
e_g_h_rgb[e_g_h < -eps] = (255, 0, 0)
e_g_h_rgb[e_g_h > eps] = (0, 0, 255)

plt.subplot(1, 3, 1)
plt.imshow(e_rgb)
plt.title("Original Event Tensor")
plt.xticks([])
plt.yticks([])
plt.axis('off')
#plt.show()

plt.subplot(1, 3, 2)
plt.imshow(e_g_rgb)
plt.title("+ Gaussian Noise")
plt.xticks([])
plt.yticks([])
plt.axis('off')
#plt.show()

plt.subplot(1, 3, 3)
plt.imshow(e_g_h_rgb)
plt.title("+ Hot Pixels")
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()
