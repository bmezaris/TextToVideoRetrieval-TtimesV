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
# from model_bert import get_model, get_we_parameter
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

    selected_captions_num = 100
    capfile = '/data/dgalanop/CERTH_VisualSearch_dualDense/TGIF_MSR_VTT_Activity_Vatex_2fps/TextData' \
              '/TGIF_MSR_VTT_Activity_Vatex_2fps.caption.txt'
    capList = [line.rstrip('\n').split(' ', 1)[1] for line in open(capfile) if
               len(line.rstrip('\n').split(' ', 1)[1].split(' ')) < 12]
    selected_captions = random.sample(capList, selected_captions_num)
    queriesFile16_17_18 = 'data/tv16_17_18.avs.topics_parsed.txt'
    lineList16_17_18 = [line.rstrip('\n') for line in open(queriesFile16_17_18)]

    queriesFile19 = 'data/tv19.avs.topics_parsed.txt'
    lineList19 = [line.rstrip('\n') for line in open(queriesFile19)]

    queriesFile20 = 'data/tv20.avs.topics_parsed.txt'
    lineList20 = [line.rstrip('\n') for line in open(queriesFile20)]

    queriesFileprogress = 'data/trecvid.progress.avs.topics_parsed.txt'
    lineListprogress = [line.rstrip('\n') for line in open(queriesFileprogress)]

    queriesFileprogress = 'data/tv21.avs.topics.txt'
    lineList21 = [line.rstrip('\n') for line in open(queriesFileprogress)]

    selected_captions = lineList16_17_18 + lineList19 + lineList20 + lineListprogress + lineList21

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

    # calculate random queries emb

    rand_cap_embeddings = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    cap_1 = []
    for cap in selected_captions:
        query = cap

        data_2 = dataLoadedText_one(query, bow2vec, rnn_vocab, tokenizer, options)
        cap_1.extend(data_2)
    captions_rand = collate_text_gru_fn(cap_1)
    #     captions = collate_text_gru_fn(data_2)
    #     # compute the embeddings
    #     vid_emb, cap_emb = model.forward_emb(videos, captions, True)
    #     if not rand_cap_embeddings:
    #         rand_cap_embeddings = list(cap_emb)
    #     else:
    #         for qq in range(len(cap_emb)):
    #             for kk in range(0, cap_emb[qq].__len__()):
    #                 rand_cap_embeddings[qq][kk] = torch.cat((rand_cap_embeddings[qq][kk], cap_emb[qq][kk]))
    # rand_cap_embeddings = tuple(rand_cap_embeddings)
    # for i in range(len(rand_cap_embeddings)):
    #     for j in range(len(rand_cap_embeddings[i])):
    #         rand_cap_embeddings[i][j] = rand_cap_embeddings[i][j].data.cpu().numpy().copy()

    opt.val_metric = "recall"
    opt.direction = 'bidir'
    validate_DS(opt, data_loaders['val'], model, captions_rand, measure=options.measure)


def validate_DS(opt, val_loader, model, captions_rand, measure='cosine'):
    # compute the encoding for all the validation video and captions
    video_embs, cap_embs, video_ids, caption_ids, cap_embs_rand = evaluation.encode_data_DS(model, val_loader, captions_rand, opt.log_step, logging.info)

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
    c2i_all_errors_rand = evaluation.cal_error_multiple(video_embs, cap_embs_rand)

    c2i_all_errors_new = np.zeros((len(c2i_all_errors), c2i_all_errors.shape[1]))
    for i in range(len(c2i_all_errors)):
        # print(i)
        quer_score = c2i_all_errors[i,:]
        z = np.concatenate((np.expand_dims(quer_score, axis=0), c2i_all_errors_rand), axis=0)
        z_2 = dual_softmax_pt(z)
        c2i_all_errors_new[i,:] = z_2[0,:]
        # if i == 0:
        #     c2i_all_errors_new = np.expand_dims(z_2[0,:], axis=0)
        # else:
        #     c2i_all_errors_new = np.concatenate((c2i_all_errors_new, np.expand_dims(z_2[0,:], axis=0)), axis=0)

    # c2i_all_errors_new =dual_softmax_pt(c2i_all_errors)
    if opt.val_metric == "recall":

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = evaluation.t2i(c2i_all_errors, n_caption=opt.n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors, n_caption=opt.n_caption)

        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * " + '-' * 10)
        print('t2i_map', t2i_map_score)

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = evaluation.t2i(c2i_all_errors_new, n_caption=opt.n_caption)
        t2i_map_score = evaluation.t2i_map(c2i_all_errors_new, n_caption=opt.n_caption)

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


def collate_text_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, tokens_tensor, segments_tensors, caption_text = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    # 'BERT Process'
    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths_bert = [len(seg) for seg in segments_tensors]
        tokens_tensor_padded = torch.zeros(len(tokens_tensor), max(lengths_bert)).long()
        segments_tensors_padded = torch.zeros(len(segments_tensors), max(lengths_bert)).long()
        words_mask_bert = torch.zeros(len(tokens_tensor), max(lengths_bert))

        for i, cap in enumerate(tokens_tensor):
            end = lengths_bert[i]
            tokens_tensor_padded[i, :end] = cap[:end]
            words_mask_bert[i, :end] = 1.0
        for i, cap in enumerate(segments_tensors):
            end = lengths_bert[i]
            segments_tensors_padded[i, :end] = cap[:end]


    else:
        lengths_bert = None
        tokens_tensor_padded = None
        segments_tensors_padded = None
        words_mask_bert = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    # CLIP_token = torch.squeeze(torch.tensor(clip.tokenize([a.decode("utf-8") for a in caption_text], truncate=True)))
    CLIP_token = torch.squeeze(clip.tokenize([a.decode("utf-8") for a in caption_text], truncate=True).clone().detach())

    if CLIP_token.size().__len__() == 1:
        CLIP_token = torch.unsqueeze(CLIP_token, dim=0)

    text_data = (
        target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert,
        caption_text, CLIP_token)

    return text_data


def dataLoadedText_one( query, bow2vec, vocab, tokenizer, options):
    data = []

    # Text encoding
    cap_tensors = []
    cap_bows = []

    caption_text = query[:]
    caption_text = ' '.join(clean_str(caption_text))
    caption_text = text2Berttext(caption_text, tokenizer)
    # caption_text = caption_text.encode("utf-8")
    caption_text = caption_text.encode("utf-8").decode("utf-8")

    if bow2vec is not None:
        cap_bow = bow2vec.mapping(caption_text)
        if cap_bow is None:
            cap_bow = torch.zeros(bow2vec.ndims)
        else:
            cap_bow = torch.Tensor(cap_bow)
    else:
        cap_bow = None

    if vocab is not None:
        tokens = clean_str(caption_text)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        cap_tensor = torch.Tensor(caption)
    else:
        cap_tensor = None

    # cap_tensors.append(cap_tensor.unsqueeze(0))
    # cap_bows.append(cap_bow.unsqueeze(0))

    # BERT
    caption_text = query[:]
    caption_text = ' '.join(clean_str(query))
    marked_text = "[CLS] " + caption_text + " [SEP]"
    # print marked_text
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segments_ids)

    caption_text = caption_text.encode("utf-8")

    data.append([cap_tensor, cap_bow, tokens_tensor, segments_tensors, caption_text])

    return data
    # return video_data, text_data




def groupc3(listtest):
    for x, y in itertools.groupby(enumerate(listtest), lambda a_b: a_b[0] - a_b[1]):
        y = list(y)
        yield y[0][1], y[-1][1]


def text2Berttext(caption_text, tokenizer):
    tokenized_text = tokenizer.tokenize(caption_text)
    retuned_tokenized_text = tokenized_text[:]
    # print caption_text
    # print tokenized_text
    res = [coun for coun, ele in enumerate(tokenized_text) if ('##' in ele)]

    res2 = list(groupc3(res))
    # print res
    # print (str(res2))

    for ree in res2:
        start = ree[0] - 1
        end_ = ree[1]
        tmp_token = ''
        for i in range(start, end_ + 1):
            # print tokenized_text[i].replace('##', '')
            tmp_token = tmp_token + tokenized_text[i].replace('##', '')
        # print tmp_token
        for i in range(start, end_ + 1):
            retuned_tokenized_text[i] = tmp_token
    # print tokenized_text
    # print retuned_tokenized_text
    return ' '.join(retuned_tokenized_text)



def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    if L2_norm == 0.0:
        L2_norm = np.finfo(float).eps
    return 1.0 * np.array(vec) / L2_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar' % best_epoch)


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dual_softmax_pt(sim_matrix):
    ss = torch.nn.Softmax(dim=0)
    m = ss(torch.tensor(sim_matrix))
    ss = torch.nn.Softmax(dim=1)
    n = ss(torch.tensor(sim_matrix))

    c = m * n
    # return (np.multiply(m.data.cpu().numpy().copy(), n.data.cpu().numpy().copy()))
    return c.data.numpy()


if __name__ == '__main__':
    main()


