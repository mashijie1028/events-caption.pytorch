import os
import numpy as np
import glob
from tqdm import tqdm
from my_utils import pretrain_utils, torchvision_models


from configs import event_config


DATA_PATH = r'./data/'
IMG_FOLDER_PATH = r'./video2images/'
#EVENT_FOLDER_PATH = r'/home/CBICR/msj/events-caption.pytorch/event_tensor_folder_1/'
EVENT_FOLDER_PATH = r'./data/msj/event_tensor_folder_1/'
#EVENT_FOLDER_PATH = r'./data/msj/event_tensor_folder_2/'

global C, H, W
C, H, W = 3, 224, 224
model = torchvision_models.resnet152(pretrained='imagenet')
load_image_fn = pretrain_utils.LoadTransformImage(model)


# setting of noise parameters
# ================================================================================
num_bins = event_config['num_bins']
n_frame_steps = event_config['n_frame_steps']
# contrast threshold (CT)
CT_mean = event_config['CT']['CT_mean']
CT_std = event_config['CT']['CT_std']
CT_min = event_config['CT']['CT_min']
gauss_mean = event_config['gauss_noise']['gauss_mean']
gauss_std = event_config['gauss_noise']['gauss_std']
hot_mean = event_config['hot_pixel']['hot_mean']
hot_std = event_config['hot_pixel']['hot_std']
hot_p = event_config['hot_pixel']['hot_p']


img_file_id_list = list(range(10000))
for img_file_id in tqdm(img_file_id_list):
    dst = os.path.join(IMG_FOLDER_PATH, 'video%d' % img_file_id)

    # instantiate CT and noises
    CT = np.random.randn(num_bins, H, W) * CT_std + CT_mean
    CT[CT < CT_min] = CT_min
    gauss_noise = np.random.randn(num_bins, H, W) * gauss_std + gauss_mean
    hot_pixel = np.random.randn(num_bins, H, W) * hot_std + hot_mean
    hot_mask = np.random.rand(num_bins, H, W)
    hot_pixel[hot_mask > hot_p] = 0.0

    image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
    samples = np.round(np.linspace(
        0, len(image_list) - 1, (num_bins + 1) * n_frame_steps))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), C, H, W))
    for iImg in range(len(image_list)):
        img = load_image_fn(image_list[iImg])
        images[iImg] = img
    images = np.mean(images, axis=1, keepdims=False)

    img_diff = np.zeros((n_frame_steps, num_bins, H, W))
    event_tensor = np.zeros((n_frame_steps, num_bins, H, W))
    for i in range(n_frame_steps):
        for j in range(num_bins):
            img_diff[i, j, :, :] = images[i * (num_bins + 1) + j + 1, :, :] - images[i * (num_bins + 1) + j, :, :]

    event_tensor[img_diff > CT] = 1.0
    event_tensor[img_diff < -CT] = -1.0
    event_tensor += gauss_noise
    event_tensor += hot_pixel

    save_path = os.path.join(EVENT_FOLDER_PATH, 'event_tensor_%d' % img_file_id + '.npy')
    np.save(save_path, event_tensor)

