"""
Author: Mashijie
Date: 2021-05-14

Forked from video caption
utils for training the video caption model and events caption model
"""

import os
import numpy as np
import torch


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N


def save_model(model, optimizer, scheduler, epoch, running_loss_list, test_loss_list, best_test_loss, isBest=False,
               save_dir=r'/home/CBICR/msj/events-caption.pytorch/checkpoints/resnet50_n_8/'):
    print('Saving checkpoints......')
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler_state_dict': scheduler.state_dict(),
                  'epoch': epoch,
                  'running_loss_list': running_loss_list,
                  'test_loss_list': test_loss_list,
                  'best_test_loss': best_test_loss
                  }
    # path_checkpoint = "./checkpoints/checkpoint_{}_epoch.pth".format(epoch)
    path_checkpoint = os.path.join(save_dir, 'EveCap_{}_epoch.pth.tar'.format(epoch + 1))   # epoch (in for) +1 = .tar epoch
    if isBest:
        path_checkpoint = os.path.join(save_dir,
                                       'EveCap_is_best_{}_epoch.pth.tar'.format(epoch + 1))  # epoch (in for) +1 = .tar epoch

    torch.save(checkpoint, path_checkpoint)


def load_model(model,  load_epoch, optimizer, scheduler, isBest=False,
               load_dir=r'/home/CBICR/msj/events-caption.pytorch/checkpoints/resnet50_n_8/'):
    print('Loading checkpoints......')
    path_checkpoint = os.path.join(load_dir, 'EveCap_{}_epoch.pth.tar'.format(load_epoch))  # epoch (in for) +1 = load_epoch

    if isBest:
        path_checkpoint = os.path.join(load_dir,
                                       'EveCap_is_best_{}_epoch.pth.tar'.format(load_epoch))  # epoch (in for) +1 = load_epoch

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    running_loss_list = checkpoint['running_loss_list']
    test_loss_list = checkpoint['test_loss_list']
    best_test_loss = checkpoint['best_test_loss']

    return {'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
            'epoch': epoch,
            'running_loss_list': running_loss_list,
            'test_loss_list': test_loss_list,
            'best_test_loss': best_test_loss
            }
