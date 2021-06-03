"""
Author: mashijie
Date: 2021-05-11
Update: 2021-05-15
Update2: 2021-05-16
update2 for pre-exported event tensor

code for dataloader of events caption
load images from image folders and convert them to event tensor (event frames)
"""


import os
import numpy as np
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import glob
from my_utils import pretrain_utils
import pretrainedmodels

from configs import event_config

DATA_PATH = r'/home/CBICR/msj/events-caption.pytorch/data/'
IMG_FOLDER_PATH = r'/home/CBICR/msj/events-caption.pytorch/video2images/'
EVENT_FOLDER_PATH = r'/data/msj/event_tensor_folder_1/'
#EVENT_FOLDER_PATH = r'/data/msj/event_tensor_folder_2/'

if EVENT_FOLDER_PATH == r'/data/msj/event_tensor_folder_1/':
    event_config['n_frame_steps'] = 8
if EVENT_FOLDER_PATH == r'/data/msj/event_tensor_folder_2/':
    event_config['n_frame_steps'] = 40

global C, H, W
C, H, W = 3, 224, 224


class EventDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode, event_config=event_config):
        super(EventDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        # print('number of val videos: ', len(self.splits['val']))
        print('number of val videos: ', len(self.splits['validate']))
        print('number of test videos: ', len(self.splits['test']))

        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

        # prepare for get event tensor
        self.model = pretrainedmodels.resnet50(pretrained='imagenet')
        self.load_image_fn = pretrain_utils.LoadTransformImage(self.model)
        self.event_config = event_config

        print('load event tensors from %s' % EVENT_FOLDER_PATH)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            # ix = ix + len(self.splits['train']) + len(self.splits['val'])
            ix = ix + len(self.splits['train']) + len(self.splits['validate'])

        # ===========================================================================================
        # get event_tensors
        # (8, 5, 224, 224) (n_frame_steps, num_bins, H, W)
        event_tensor_np = np.load(os.path.join(EVENT_FOLDER_PATH, 'event_tensor_%d' % ix + '.npy'))
        event_tensor = torch.from_numpy(event_tensor_np)

        # ===========================================================================================
        # get labels, masks, gts and video_ids
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions['video%i' % (ix)]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        # data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['event_tensor'] = event_tensor.type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'video%i' % (ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])
