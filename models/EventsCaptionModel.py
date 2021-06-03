"""
Author: mashijie
Date: 2021-05-11

Update: 2021-05-16 for exported event tensors
update the code in EventsCaptionModel.forward()

Total Model for events caption: CNN + LSTM
CNN: for feature extraction
LSTM: for text generation
"""

import torch
import torch.nn as nn
import torchvision


class EventsCaptionModel0(nn.Module):
    def __init__(self, cnn, encoder, decoder, dim_vid=2048):
        """

        :param cnn:
        :param encoder: encoder in LSTM
        :param decoder: encoder in LSTM
        """
        super(EventsCaptionModel0, self).__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.dim_vid = dim_vid

    def forward(self, event_tensor, target_variable=None,
                mode='train', opt={}):
        """

        Args:
            event_tensor (Variable): input event tensor of shape [batch_size, n_frame_steps, num_bins, H, W]
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        batch_size, n_frame_steps, num_bins, H, W = event_tensor.size()
        vid_feats = torch.zeros([batch_size, n_frame_steps, self.dim_vid], requires_grad=True).cuda()

        for i in range(n_frame_steps):
            vid_feats[:, i, :] = self.cnn(event_tensor[:, i, :, :, :]).squeeze()

        vid_feats.requires_grad_(True)
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds


