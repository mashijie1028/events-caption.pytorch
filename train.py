"""
Author: Mashijie
Date: 2021-05-15
Update: 2021-05-16

code for training events caption model: CNN + LSTM (jointly)
partially forked from video caption training code by Mashijie

"""

# general import for video caption and events caption
# ==================================================================================
import json
import os
import time
import numpy as np

import misc.utils as misc_utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from event_dataloader import EventDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader

from my_utils.utils import count_network_parameters, save_model, load_model

# specialities import for event caption
# ==================================================================================
import torchvision
import my_utils.pretrain_utils as pretrain_utils
from configs import event_config
from models.EventsCaptionModel import EventsCaptionModel0


# update for pytorch 1.6
# cnn model: resnet 50 by default

DATA_PATH = r'/home/CBICR/msj/events-caption.pytorch/data/'
IMG_FOLDER_PATH = r'/home/CBICR/msj/events-caption.pytorch/video2images/'
EVENT_FOLDER_PATH = r'/data/msj/event_tensor_folder_1/'
#EVENT_FOLDER_PATH = r'/data/msj/event_tensor_folder_2/'

if EVENT_FOLDER_PATH == r'/data/msj/event_tensor_folder_1/':
    event_config['n_frame_steps'] = 8
if EVENT_FOLDER_PATH == r'/data/msj/event_tensor_folder_2/':
    event_config['n_frame_steps'] = 40


opt = opts.parse_opt()
opt = vars(opt)
opt["caption_json"] = r'data/total_caption.json'
opt["input_json"] = r'data/train_val_videodatainfo.json'
opt["info_json"] = r'data/total_info.json'
# opt["feats_dir"] = [r'data/feats/resnet50/']

opt["save_checkpoint_every"] = 20

# opt["checkpoint_path"] = r'data/save_total'
os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
if not os.path.isdir(opt["checkpoint_path"]):
    os.mkdir(opt["checkpoint_path"])
with open(opt_json, 'w') as f:
    json.dump(opt, f)
print('save opt details to %s' % opt_json)

train_dataset = EventDataset(opt, 'train', event_config=event_config)
train_loader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)
test_dataset = EventDataset(opt, 'test', event_config=event_config)
test_loader = DataLoader(test_dataset, batch_size=opt["batch_size"], shuffle=False)
opt["vocab_size"] = train_dataset.get_vocab_size()


# init feature extractor (CNN) model
# ==================================================================================
resnet50 = torchvision.models.resnet50(pretrained=True)
print(resnet50.inplanes)
resnet50.conv1 = nn.Conv2d(event_config['num_bins'], 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
resnet50.fc = pretrain_utils.Identity()


# init language (LSTM) model
# ==================================================================================
if opt["model"] == 'S2VTModel':
    language_model = S2VTModel(
        opt["vocab_size"],
        opt["max_len"],
        opt["dim_hidden"],
        opt["dim_word"],
        opt['dim_vid'],
        rnn_cell=opt['rnn_type'],
        n_layers=opt['num_layers'],
        rnn_dropout_p=opt["rnn_dropout_p"])
elif opt["model"] == "S2VTAttModel":
    encoder = EncoderRNN(
        opt["dim_vid"],
        opt["dim_hidden"],
        bidirectional=opt["bidirectional"],
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(
        opt["vocab_size"],
        opt["max_len"],
        opt["dim_hidden"],
        opt["dim_word"],
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"],
        bidirectional=opt["bidirectional"])
    language_model = S2VTAttModel(encoder, decoder)


# total model: events caption model
# ==================================================================================
#model = EventsCaptionModel(cnn=resnet50, encoder=encoder, decoder=decoder, dim_vid=opt["dim_vid"])
model = EventsCaptionModel0(resnet50, encoder, decoder, opt["dim_vid"])
model = model.cuda()
model = nn.DataParallel(model)

num_params_cnn = count_network_parameters(resnet50)
print('\n=====================================================================')
print("===> CNN Model has %d parameters" % num_params_cnn)
print('=====================================================================')

num_params_language = count_network_parameters(language_model)
print('\n=====================================================================')
print("===> Language Model has %d parameters" % num_params_language)
print('=====================================================================')

num_params_total = count_network_parameters(model)
print('\n=====================================================================')
print("===> Total Events Caption Model has %d parameters" % num_params_total)
print('=====================================================================')


crit = misc_utils.LanguageModelCriterion()
rl_crit = misc_utils.RewardCriterion()
optimizer = optim.Adam(
    model.parameters(),
    lr=opt["learning_rate"],
    weight_decay=opt["weight_decay"])
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=opt["learning_rate_decay_every"],
    gamma=opt["learning_rate_decay_rate"])
lr_scheduler = exp_lr_scheduler

begin_epoch = -1
total_epoch = opt["epochs"]
running_loss_list = []
test_loss_list = []
best_test_loss = 1e6

# load data
isLoad = False
if isLoad:
    # load_epoch: name of .tar
    ckpt = load_model(model=model, load_epoch=1, optimizer=optimizer, scheduler=lr_scheduler,
                      load_dir=opt["checkpoint_path"])
    model = ckpt['model']
    optimizer = ckpt['optimizer']
    scheduler = ckpt['scheduler']
    begin_epoch = ckpt['epoch']
    running_loss_list = ckpt['running_loss_list']
    test_loss_list = ckpt['test_loss_list']
    best_test_loss = ckpt['best_test_loss']

# train and test for events caption
print("\n\n\n=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("Start Training!\n")
with open("video_caption_screen_log.txt", "w") as f1:
    for epoch in range(begin_epoch + 1, total_epoch):
        print('\n\nEpoch: %d' % (epoch + 1))
        print("=====================================================================")
        print("Begin Training!")
        model.train()
        start_time = time.time()

        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            torch.cuda.synchronize()
            event_tensor = data['event_tensor'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, _ = model(event_tensor, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(
                    event_tensor, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, event_tensor, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()

            # train_loss = loss.item()
            running_loss += loss.item()
            torch.cuda.synchronize()

            # if not sc_flag:
            #     print("iter %d (epoch %d), train_loss = %.6f" %
            #           (iteration, epoch, train_loss))
            # else:
            #     print("iter %d (epoch %d), avg_reward = %.6f" %
            #           (iteration, epoch, np.mean(reward[:, 0])))

            if batch_idx % 50 == 49 or batch_idx == len(train_loader) - 1:
                if batch_idx == len(train_loader) - 1:
                    running_loss *= 50 / (len(train_loader) - len(train_loader) // 50 * 50)
                print('Epoch [%4d/%4d], Step [%3d/%3d], running_loss: %.06f'
                      % (epoch + 1, total_epoch, batch_idx + 1, len(train_loader),
                         running_loss / 50))
                elapsed_time = time.time() - start_time
                print('Time elapsed: %.5f' % elapsed_time)

                f1.write('Epoch [%4d/%4d], Step [%3d/%3d], running_loss: %.06f'
                         % (epoch + 1, total_epoch, batch_idx + 1, len(train_loader),
                            running_loss / 50))
                f1.write('\n')
                f1.write('Time elapsed: %.5f' % elapsed_time)
                f1.write('\n')

                running_loss_list.append(running_loss / 50)
                running_loss = 0.0

        print("\n=====================================================================")
        print("Begin Evaluation!")
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch_idx, data in enumerate(test_loader):
                torch.cuda.synchronize()
                event_tensor = data['event_tensor'].cuda()
                labels = data['labels'].cuda()
                masks = data['masks'].cuda()

                if not sc_flag:
                    seq_probs, _ = model(event_tensor, labels, 'train')
                    loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
                else:
                    seq_probs, seq_preds = model(
                        event_tensor, mode='inference', opt=opt)
                    reward = get_self_critical_reward(model, event_tensor, data,
                                                      seq_preds)
                    print(reward.shape)
                    loss = rl_crit(seq_probs, seq_preds,
                                   torch.from_numpy(reward).float().cuda())

                clip_grad_value_(model.parameters(), opt['grad_clip'])

                test_loss += loss.item()
                torch.cuda.synchronize()

            test_loss /= len(test_loader)
            print('test_loss: %.06f' % test_loss)
            elapsed_time = time.time() - start_time
            print('Time elapsed: %.5f' % elapsed_time)
            f1.write('test_loss: %.06f' % test_loss)
            f1.write('\n')
            f1.write('Time elapsed: %.5f' % elapsed_time)
            f1.write('\n')
            f1.flush()

            test_loss_list.append(test_loss)

            if test_loss < best_test_loss:
                print("Congrats, best results!")
                f2 = open("best_test_loss.txt", "w")
                f2.write("EPOCH=%d, best_test_loss= %.6f" % (epoch + 1, test_loss))
                f2.close()
                best_test_loss = test_loss

                if epoch > 59:
                    save_model(model=model, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch,
                               running_loss_list=running_loss_list, test_loss_list=test_loss_list,
                               best_test_loss=best_test_loss,
                               isBest=True, save_dir=opt["checkpoint_path"])

                # if not sc_flag:
                #     print("iter %d (epoch %d), train_loss = %.6f" %
                #           (iteration, epoch, train_loss))
                # else:
                #     print("iter %d (epoch %d), avg_reward = %.6f" %
                #           (iteration, epoch, np.mean(reward[:, 0])))

        # learning rate tuning
        lr_scheduler.step()

        # save model checkpoints
        if epoch == 0 or (epoch + 1) % opt["save_checkpoint_every"] == 0 or epoch == total_epoch - 1:
            save_model(model=model, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch,
                       running_loss_list=running_loss_list, test_loss_list=test_loss_list,
                       best_test_loss=best_test_loss, save_dir=opt["checkpoint_path"])

        # tmp save loss lists per opt["save_checkpoint_every"] epochs
        if epoch % opt["save_checkpoint_every"] == opt["save_checkpoint_every"] - 1:
            f3 = open(os.path.join(opt["checkpoint_path"], "tmp_record_losses_lists.txt"), "w")
            f3.write("running_loss_list: \n")
            f3.write(str(running_loss_list))
            f3.write("\n")
            f3.write("test_loss_list: \n")
            f3.write(str(test_loss_list))
            f3.write("\n")
            f3.close()

        # if epoch % opt["save_checkpoint_every"] == 0:
        #     model_path = os.path.join(opt["checkpoint_path"],
        #                               'model_%d.pth' % (epoch))
        #     model_info_path = os.path.join(opt["checkpoint_path"],
        #                                    'model_score.txt')
        #     torch.save(model.state_dict(), model_path)
        #     print("model saved to %s" % (model_path))
        #     with open(model_info_path, 'a') as f:
        #         f.write("model_%d, loss: %.6f\n" % (epoch, ))

print("\n\n\n=====================================================================")
print("=====================================================================")
print("=====================================================================")
print('End of training...')

f4 = open("record_losses_lists.txt", "w")
f4.write("running_loss_list: \n")
f4.write(str(running_loss_list))
f4.write("\n")
f4.write("test_loss_list: \n")
f4.write(str(test_loss_list))
f4.write("\n")
f4.close()
