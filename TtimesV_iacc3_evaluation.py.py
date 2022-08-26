from __future__ import print_function
import os
import sys
import time

import torch

from TxV_models.model_TtimesV import get_model, get_we_parameter
from util.text2vec import get_text_encoder

import logging
import json
import numpy as np
import pickle

import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from scipy.spatial import distance
from util.vocab import clean_str
from util.vocab import Vocabulary

from pytorch_transformers import BertTokenizer
import itertools
import AVS_datasetload as dataload
import random
import clip
import random

random.seed(10)

import pickle

VIDEO_MAX_LEN = 64


def clip_encoding(CLIP_model, queries):
    CLIP_token = torch.squeeze(clip.tokenize([a.decode("utf-8") for a in queries], truncate=True).clone().detach())
    with torch.no_grad():
        CLIP_features = CLIP_model.encode_text(CLIP_token).float()
        
    return CLIP_features


def dual_softmax_pt(sim_matrix):
    ss = torch.nn.Softmax(dim=0)
    m = ss(torch.tensor(sim_matrix))
    ss = torch.nn.Softmax(dim=1)
    n = ss(torch.tensor(sim_matrix))

    c = m * n

    return c.data.numpy()


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    if L2_norm == 0.0:
        L2_norm = np.finfo(float).eps
    return 1.0 * np.array(vec) / L2_norm


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


def cosine_calculate(matrix_a, matrix_b):
    result = distance.cdist(matrix_a, matrix_b, 'cosine')
    # return result.tolist()
    return result


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
    return ' '.join(retuned_tokenized_text)


def error_calulator_v2(videos, captions, errtype='sum'):
    """ Caption vectors are numpy arrays instead of pytorch tensors
        Video vectors are numpy arrays instead of pytorch tensors
    """
    if errtype == 'sum':
        errors = np.zeros((len(videos[0][0]), len(captions[0][0])))
        for i in range(len(captions)):
            for j in range(len(videos)):
                capt = l2norm(captions[i][j])
                vid = l2norm(videos[j][i])
                errors += cosine_calculate(vid, capt)
        return errors
    elif errtype == 'max':
        capt = l2norm(captions[0][0])
        vid = l2norm(videos[0][0])
        errors = cosine_calculate(vid, capt).tolist()
        for i in range(1, len(captions)):
            capt = l2norm(captions[i])
            vid = l2norm(videos[i])
            err = cosine_calculate(vid, capt).tolist()
            cc = []
            for aa, bb in zip(errors, err):
                cc.append([max(els) for els in zip(aa, bb)])
            errors = cc.copy()

        return errors
    elif errtype == 'min':
        capt = l2norm(captions[0])
        vid = l2norm(videos[0])
        errors = cosine_calculate(vid, capt).tolist()
        for i in range(1, len(captions)):
            capt = l2norm(captions[i])
            vid = l2norm(videos[i])
            err = cosine_calculate(vid, capt).tolist()
            cc = []
            for aa, bb in zip(errors, err):
                cc.append([min(els) for els in zip(aa, bb)])
            errors = cc.copy()

        return errors


def check(resultFile, pattern):
    with open(resultFile) as f:
        datafile = f.readlines()
    for line in datafile:
        if pattern in line:
            print(line.rstrip("\n\r"))


def dataLoadedVideoText_one(video2frames, video_id, visual_feats, query, bow2vec, vocab, tokenizer, options):
    data = []

    videos = []

    frame_list = video2frames[video_id]
    frame_vecs = []
    frames_tensors = []
    for vis_fea in visual_feats:
        frame_vecs = []
        for frame_id in frame_list:
            # l_2 normalize
            if (options.do_visual_feas_norm):
                frame_vecs.append(do_L2_norm(visual_feats[vis_fea].read_one(frame_id)))
            else:
                frame_vecs.append(visual_feats[vis_fea].read_one(frame_id))
        frames_tensor = torch.Tensor(np.array(frame_vecs))
        frames_tensors.append(frames_tensor)

    # Text encoding
    cap_tensors = []
    cap_bows = []

    caption_text = query[:]
    caption_text = ' '.join(clean_str(caption_text))
    caption_text = text2Berttext(caption_text, tokenizer)
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

    data.append([frames_tensors, cap_tensor, cap_bow, tokens_tensor, segments_tensors, caption_text])

    return data


def dataLoadedText_one(query, bow2vec, vocab, tokenizer, options):
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


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows, tokens_tensor, segments_tensors, caption_text = zip(*data)

    num_of_feas = len(videos[0])

    vidoes_all = []
    videos_origin_all = []
    video_lengths_all = []
    vidoes_mask_all = []
    for fea in range(num_of_feas):
        frame_vec_len = len(videos[0][fea][0])
        video_lengths = [min(VIDEO_MAX_LEN, len(frame[0])) for frame in videos]
        vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
        videos_origin = torch.zeros(len(videos), frame_vec_len)
        vidoes_mask = torch.zeros(len(videos), max(video_lengths))

        for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[fea][:end, :]
            # Fil the zeros of vidoes with random frames
            if end < max(video_lengths):
                try:
                    num_of_filler_frames = random.choices(list(range(0, end)), k=(max(video_lengths) - end))
                    # print(num_of_filler_frames)
                    # print()
                except:
                    print()
                vidoes[i, end:, :] = frames[fea][num_of_filler_frames, :]
            videos_origin[i, :] = torch.mean(frames[fea], 0)
            vidoes_mask[i, :end] = 1.0

        vidoes_all.append(vidoes)
        videos_origin_all.append(videos_origin)
        video_lengths_all.append(video_lengths)
        vidoes_mask_all.append(vidoes_mask)

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

    video_data = (vidoes_all, videos_origin_all, video_lengths, vidoes_mask)
    text_data = (
        target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert,
        caption_text, CLIP_token)

    return video_data, text_data


def collate_text_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)

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

    text_data = (
        target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert,
        caption_text, CLIP_token)

    return text_data


def parse_args():
    # Hyper Parameters
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
    testCollection = opt.testCollection
    # n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    batch_size = opt.batch_size

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    opt.dual_softmax = 0
    opt.dual_softmax_same_dataset = 0
    saveFile_AVS16 = (opt.logger_name + '/AVS16_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS17 = (opt.logger_name + '/AVS17_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS18 = (opt.logger_name + '/AVS18_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')

    if os.path.exists(saveFile_AVS18) & (opt.overwrite == 0):
        sys.exit(0)

    queriesFile = 'data/tv16_17_18.avs.topics_parsed.txt'
    lineList = [line.rstrip('\n') for line in open(queriesFile)]

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

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection)
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    # data loader prepare
    # caption_files = {'test': os.path.join(evalpath, testCollection, 'TextData', '%s.caption.txt' % testCollection)}
    options.visual_features = options.visual_feature.split('@')
    visual_feat_path = {y: os.path.join(evalpath, testCollection, 'FeatureData', y)
                        for y in options.visual_features}
    visual_feats = {'test': {y: BigFile(visual_feat_path[y]) for y in options.visual_features}}

    assert options.visual_feat_dim == [visual_feats['test'][aa].ndims for aa in visual_feats['test']]
    video2frames = {'test': read_dict(
        os.path.join(evalpath, testCollection, 'FeatureData', options.visual_features[0], 'video2frames.txt'))}
    # video2frames = None

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow',
                                  options.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn',
                                  options.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # initialize word embedding
    options.we_parameter = None
    if options.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        options.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    # switch to evaluate mode
    model.val_start()

    video2frames = video2frames['test']
    videoIDs = [key for key in video2frames.keys()]

    # Queries embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    queryEmbeddings = []
    for quer in lineList:
        query = quer
        videBatch = videoIDs[0]  # a dummy video
        data = dataLoadedVideoText_one(video2frames, videBatch, visual_feats['test'], query, bow2vec, rnn_vocab,
                                       tokenizer, options)
        videos, captions = collate_frame_gru_fn(data)
        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)
        if not queryEmbeddings:
            queryEmbeddings = list(cap_emb)
        else:
            for qq in range(len(cap_emb)):
                for kk in range(0, cap_emb[qq].__len__()):
                    queryEmbeddings[qq][kk] = torch.cat((queryEmbeddings[qq][kk], cap_emb[qq][kk]))
    queryEmbeddings = tuple(queryEmbeddings)
    for i in range(len(queryEmbeddings)):
        for j in range(len(queryEmbeddings[i])):
            queryEmbeddings[i][j] = queryEmbeddings[i][j].data.cpu().numpy().copy()

    # Dummy caption data in batch size
    data = []
    for o in range(batch_size):
        data.extend(dataLoadedText_one(query, bow2vec, rnn_vocab, tokenizer, options))
    captions = collate_text_gru_fn(data)

    data_loader = dataload.get_test_data_loaders(visual_feats['test'], batch_size, 5, options.do_visual_feas_norm,
                                                 video2frames=video2frames)
    start = time.time()
    errorlistList = []
    VideoIDS = []
    video_ids = []
    for i, (videos, idxs, vid_ids) in enumerate(data_loader):

        video_ids.extend(vid_ids)
        VideoIDS.extend(vid_ids)
        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)
        # convert vid_emb from tensor to numpy
        for ii in range(len(vid_emb)):
            for j in range(len(vid_emb[ii])):
                vid_emb[ii][j] = vid_emb[ii][j].data.cpu().numpy().copy()

        errorlistList.extend(error_calulator_v2(vid_emb, queryEmbeddings, errtype=opt.errtype))

        if errorlistList.__len__() % (batch_size * 1000) == 0:
            # print (i)
            end = time.time()
            print(str(errorlistList.__len__()) + '/' + str(len(videoIDs)) + ' in: ' + str(end - start))
            start = time.time()

    errorlist = np.asarray(errorlistList)
    np.save(opt.logger_name + '/AVS_16_17_18_errorlist.npy', errorlist)
    file_to_store = open(opt.logger_name + '/AVS_16_17_18_VideoIDS', "wb")
    pickle.dump(VideoIDS, file_to_store)
    query_errors_sofmax = dual_softmax_pt(errorlist)

    # No dual softmax inference
    opt.dual_softmax = 0
    opt.dual_softmax_same_dataset = 0
    opt.dual_softmax_same_dataset_otherAVS = 0

    saveFile_AVS16 = (opt.logger_name + '/AVS16_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS17 = (opt.logger_name + '/AVS17_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS18 = (opt.logger_name + '/AVS18_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')

    evaluation(saveFile_AVS16, saveFile_AVS17, saveFile_AVS18, lineList, VideoIDS, errorlist, query_errors_sofmax, opt)

    # Dual softmax inference using AVS queries
    opt.dual_softmax = 1
    opt.dual_softmax_same_dataset = 1
    opt.dual_softmax_same_dataset_otherAVS = 0

    saveFile_AVS16 = (opt.logger_name + '/AVS16_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS17 = (opt.logger_name + '/AVS17_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')
    saveFile_AVS18 = (opt.logger_name + '/AVS18_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '.txt')

    evaluation(saveFile_AVS16, saveFile_AVS17, saveFile_AVS18, lineList, VideoIDS, errorlist, query_errors_sofmax, opt)

    # Dual softmax inference using AVS queries from other years
    opt.dual_softmax = 1
    opt.dual_softmax_same_dataset = 1
    opt.dual_softmax_same_dataset_otherAVS = 1

    saveFile_AVS16 = (opt.logger_name + '/AVS16_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '_otherAVS_' + str(opt.dual_softmax_same_dataset_otherAVS) + '.txt')
    saveFile_AVS17 = (opt.logger_name + '/AVS17_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '_otherAVS_' + str(opt.dual_softmax_same_dataset_otherAVS) + '.txt')
    saveFile_AVS18 = (opt.logger_name + '/AVS18_' + testCollection + '_doSoftmax_' + str(opt.dual_softmax) + '_same_dataset_' + str(opt.dual_softmax_same_dataset) + '_otherAVS_' + str(opt.dual_softmax_same_dataset_otherAVS) + '.txt')

    evaluation(saveFile_AVS16, saveFile_AVS17, saveFile_AVS18, lineList, VideoIDS, errorlist, query_errors_sofmax, opt)


def evaluation(saveFile_AVS16, saveFile_AVS17, saveFile_AVS18, queriesList, VideoIDS, errorlist, query_errors_sofmax, opt):
    # AVS 2016
    print('AVS 2016')
    f = open(saveFile_AVS16, "w")
    # print("Loading query #", end=' ')
    for num, name in enumerate(queriesList[:30], start=1):
        # print(num, end=' ')
        queryError = errorlist[:, num - 1]
        if opt.dual_softmax:
            if opt.dual_softmax_same_dataset:
                if not opt.dual_softmax_same_dataset_otherAVS:
                    query_errors = query_errors_sofmax
                    queryError = query_errors[:, num - 1]
                else:
                    query_error = np.expand_dims(errorlist[:, num - 1], axis=1)
                    otherAVS_errors = errorlist[:, 30:]
                    queryError = dual_softmax_pt(np.concatenate((query_error, otherAVS_errors), axis=1))
                    queryError = queryError[:, 0]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS16, "a")
        c = 0
        existings = []
        # for ind in scoresIndex[::-1]:
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(10000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()

    resultAVSFile16 = saveFile_AVS16[:-4] + '_results.txt'
    command = "perl data/AVS/sample_eval.pl -q data/AVS/avs.qrels.tv16 {} > {}".format(saveFile_AVS16, resultAVSFile16)
    os.system(command)
    check(resultAVSFile16, 'infAP		all')

    # AVS17
    print('AVS 2017')
    f = open(saveFile_AVS17, "w")
    # print("Loading query #", end=' ')
    for num, name in enumerate(queriesList[30:60], start=31):
        # print(num, end=' ')
        queryError = errorlist[:, num - 1]
        if opt.dual_softmax:
            if opt.dual_softmax_same_dataset:
                if not opt.dual_softmax_same_dataset_otherAVS:
                    query_errors = query_errors_sofmax
                    queryError = query_errors[:, num - 1]
                else:
                    query_error = np.expand_dims(errorlist[:, num - 1], axis=1)
                    otherAVS_errors = np.concatenate((errorlist[:, :30], errorlist[:, 60:]), axis=1)

                    queryError = dual_softmax_pt(np.concatenate((query_error, otherAVS_errors), axis=1))
                    queryError = queryError[:, 0]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS17, "a")
        c = 0
        existings = []
        # for ind in scoresIndex[::-1]:
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(10000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()

    resultAVSFile17 = saveFile_AVS17[:-4] + '_results.txt'
    command = "perl data/AVS/sample_eval.pl -q data/AVS/avs.qrels.tv17 {} > {}".format(saveFile_AVS17, resultAVSFile17)
    os.system(command)
    check(resultAVSFile17, 'infAP		all')

    # AVS18
    print('AVS 2018')
    f = open(saveFile_AVS18, "w")
    # print("Loading query #", end=' ')
    for num, name in enumerate(queriesList[60:90], start=61):
        # print(num, end=' ')
        queryError = errorlist[:, num - 1]
        if opt.dual_softmax:
            if opt.dual_softmax_same_dataset:
                if not opt.dual_softmax_same_dataset_otherAVS:
                    query_errors = query_errors_sofmax
                    queryError = query_errors[:, num - 1]
                else:
                    query_error = np.expand_dims(errorlist[:, num - 1], axis=1)
                    otherAVS_errors = errorlist[:, :60]

                    queryError = dual_softmax_pt(np.concatenate((query_error, otherAVS_errors), axis=1))
                    queryError = queryError[:, 0]
        scoresIndex = np.argsort(queryError)

        f = open(saveFile_AVS18, "a")
        c = 0
        existings = []
        # for ind in scoresIndex[::-1]:
        for ind in scoresIndex:
            imgID = VideoIDS[ind]
            c = c + 1
            f.write('15%02d' % num)
            f.write(' 0 ' + imgID + ' ' + str(c) + ' ' + str(10000 - c) + ' ITI-CERTH' + '\n')
            if c == 1000:
                break
    f.close()
    
    resultAVSFile18 = saveFile_AVS18[:-4] + '_results.txt'
    command = "perl data/AVS/sample_eval.pl -q data/AVS/avs.qrels.tv18 {} > {}".format(saveFile_AVS18, resultAVSFile18)
    os.system(command)
    check(resultAVSFile18, 'infAP		all')
    print()


if __name__ == '__main__':
    main()
