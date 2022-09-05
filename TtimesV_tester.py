from __future__ import print_function
import pickle
import os
import sys
import time
import shutil
import json
import numpy as np
from util.vocab import clean_str
import itertools
import random
from pytorch_transformers import BertTokenizer
import clip

import TtimesV_evaluation as evaluation

import util.data_provider_TtimesV as data

import torch
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

from TxV_models.model_TtimesV import get_model, get_we_parameter

import logging
import tensorboard_logger as tb_logger

import argparse

from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.util import read_dict, AverageMeter, LogCollector
from basic.generic_utils import Progbar

VIDEO_MAX_LEN = 64
INFO = __file__


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('--evalpath', type=str, default=ROOT_PATH,
                        help='path to evaluation video features. (default: %s)' % ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str,
                        help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video (default: 1)')
    parser.add_argument('--errtype', type=str, default='sum', choices=['sum', 'max', 'min'],
                        help='overwrite existed file. (default: 0)')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))


    rootpath = opt.rootpath
    evalpath = opt.evalpath
    valCollection = opt.testCollection


    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    batch_size = opt.batch_size


    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']

    if not hasattr(options, 'do_visual_feas_norm'):
        setattr(options, "do_visual_feas_norm", 0)

    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    # collections: trian, val
    trainCollection = options.trainCollection
    collections = {'train': trainCollection, 'val': valCollection}
    cap_file = {'train': '%s.caption.txt' % trainCollection,
                'val': '%s.caption.txt' % valCollection}
    # caption
    caption_files = {x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                     for x in collections}

    # Load visual features
    opt.visual_features = options.visual_feature.split('@')
    visual_feat_path = {y: {x: os.path.join(rootpath, collections[x], 'FeatureData', y)
                            for x in collections} for y in opt.visual_features}
    visual_feats = {'train': {y: BigFile(visual_feat_path[y]['train']) for y in opt.visual_features}}
    visual_feats['val'] = {y: BigFile(visual_feat_path[y]['val']) for y in opt.visual_features}
    opt.visual_feat_dim = [visual_feats['train'][aa].ndims for aa in visual_feats['train']]

    # visual_feats['val'] = {y: BigFile(visual_feat_path[y]['val']) for y in opt.visual_features}
    opt.visual_feat_dim = [visual_feats['val'][aa].ndims for aa in visual_feats['val']]
    trainCollection = options.trainCollection

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow',
                                  options.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # initialize word embedding
    options.we_parameter = None
    if options.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        options.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)


    # set data loader
    video2frames = {x: read_dict(os.path.join(rootpath, collections[x], 'FeatureData', options.visual_features[0], 'video2frames.txt')) for x in collections}
    data_loaders = data.get_data_loaders(
        caption_files, visual_feats, rnn_vocab, bow2vec, options.batch_size, options.workers, opt.n_caption, options.do_visual_feas_norm,
        video2frames=video2frames)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    # switch to evaluate mode
    model.val_start()

    opt.val_metric = "recall"
    opt.direction = 'bidir'
    validate(opt, data_loaders['val'], model, measure=options.measure)



def validate(opt, val_loader, model, measure='cosine'):
    # compute the encoding for all the validation video and captions
    video_embs, cap_embs, video_ids, caption_ids = evaluation.encode_data(model, val_loader, opt.log_step, logging.info)

    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)

    for ii in range(0, video_embs.__len__()):
        for kk in range(0, video_embs[ii].__len__()):
            video_embs[ii][kk] = video_embs[ii][kk][feature_mask]
    video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]

    # c2i_all_errors = evaluation.cal_error(video_embs, cap_embs, measure)
    # c2i_all_errors = evaluation.compute_multi_space_similarity(cap_embs, video_embs)
    c2i_all_errors = evaluation.cal_error_multiple(video_embs, cap_embs)
    if opt.val_metric == "recall":

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = evaluation.t2i(c2i_all_errors, n_caption=opt.n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors, n_caption=opt.n_caption)

        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * " + '-' * 10)
        print('t2i_map', t2i_map_score)

        # caption retrieval
        (r1, r5, r10, medr, meanr) = evaluation.i2t(c2i_all_errors, n_caption=opt.n_caption)
        i2t_map_score = evaluation.i2t_map(c2i_all_errors, n_caption=opt.n_caption)

        print(" * Video to text:")
        print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
        print(" * medr, meanr: {}".format([round(medr, 3), round(meanr, 3)]))
        print(" * " + '-' * 10)
        print('i2t_map', i2t_map_score)




    elif opt.val_metric == "map":
        i2t_map_score = evaluation.i2t_map(c2i_all_errors, n_caption=opt.n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors, n_caption=opt.n_caption)

        print('i2t_map', i2t_map_score)
        print('t2i_map', t2i_map_score)

    currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all' or opt.direction == 'bidir':
            currscore += (r1 + r5 + r10)
        if opt.direction == 't2i' or opt.direction == 'all' or opt.direction == 'bidir':
            currscore += (r1i + r5i + r10i)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all' or opt.direction == 'bidir':
            currscore += i2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all' or opt.direction == 'bidir':
            currscore += t2i_map_score


    return currscore


if __name__ == '__main__':
    main()

