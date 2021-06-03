"""
Author: Mashijie
Date: 2021-05-18

eval code for events caption
CNN Model: ResNet 50
Language Model: S2VT Attention Model

eval code corresponding to train.py
"""


import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from event_dataloader import EventDataset
import misc.utils as misc_utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize

# specialities import for event caption
# ==================================================================================
import torchvision
import my_utils.pretrain_utils as pretrain_utils
from configs import event_config
from models.EventsCaptionModel import EventsCaptionModel0


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def test(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in loader:
        # forward the model to get loss
        # fc_feats = data['fc_feats'].cuda()
        event_tensor = data['event_tensor'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                event_tensor, mode='inference', opt=opt)

        sents = misc_utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)


def main(opt):
    dataset = EventDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        language_model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=opt["bidirectional"],
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=opt["bidirectional"])
        language_model = S2VTAttModel(encoder, decoder).cuda()
    # model = nn.DataParallel(model)
    # Setup the model
    # model.load_state_dict(torch.load(opt["saved_model"]))

    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.conv1 = nn.Conv2d(event_config['num_bins'], 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    resnet50.fc = pretrain_utils.Identity()
    model = EventsCaptionModel0(resnet50, encoder, decoder, opt["dim_vid"])
    model = model.cuda()
    model = nn.DataParallel(model)

    checkpoint = torch.load(opt["saved_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    crit = misc_utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='',
                        help='path to saved model to evaluate')

    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v

    opt["input_json"] = r"data/test_videodatainfo.json"

    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)
