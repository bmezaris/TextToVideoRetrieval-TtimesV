from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
from scipy.spatial import distance
import torch
# from torch.autograd import Variable
from basic.metric import getScorer
from basic.util import AverageMeter, LogCollector
from scipy.spatial import distance


def cosine_calculate(matrix_a, matrix_b):
    result = distance.cdist(matrix_a, matrix_b, 'cosine')
    return result.tolist()


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


def cosine_sim(query, retrio):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return query.mm(retrio.t())


def cosine_sim_np(query_embs, retro_embs):
    query_embs = l2norm(query_embs)
    retro_embs = l2norm(retro_embs)

    return query_embs.dot(retro_embs.T)


def compute_multi_space_similarity(txt_embs:tuple, vis_embs:tuple):
    assert len(txt_embs) == len(vis_embs)

    scores = cosine_sim_np(txt_embs[0], vis_embs[0])
    for i in range(1, len(txt_embs)):
        scores += cosine_sim_np(txt_embs[i], vis_embs[i])

    return scores


def cal_error_multiple(videos, captions, measure='cosine'):
    if measure == 'cosine':
        errors = np.zeros((len(captions[0][0]), len(videos[0][0])))
        for i in range(len(captions)):
            for j in range(len(videos)):
                capt = l2norm(captions[i][j])
                vid = l2norm(videos[j][i])
                errors += -1 * numpy.dot(capt, vid.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors


def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        capt = l2norm(captions[0])
        vid = l2norm(videos[0])
        errors = -1 * numpy.dot(capt, vid.T)
        for i in range(1, len(captions)):
            capt = l2norm(captions[i])
            vid = l2norm(videos[i])
            errors += -1 * numpy.dot(capt, vid.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors


def encode_data_DS(model, data_loader, captions_rand, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    video_ids = [''] * len(data_loader.dataset)
    caption_ids = [''] * len(data_loader.dataset)
    for i, (videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, *(cap_ids, vid_ids))

        if i==0:
            vid_emb_2, cap_emb_rand = model.forward_emb(videos, captions_rand)
            cap_embs_rand = []
            for ii in range(0, cap_emb_rand.__len__()):
                cap_embs_rand_2 = []
                for kk in range(0, cap_emb_rand[ii].__len__()):
                    cap_embs_rand_2.append(np.zeros((len(cap_emb_rand[0][0]), cap_emb_rand[0][0].size(1))))
                cap_embs_rand.append(cap_embs_rand_2)

            for ii in range(0, cap_emb_rand.__len__()):
                for kk in range(0, cap_emb_rand[ii].__len__()):
                    cap_embs_rand[ii][kk] = cap_emb_rand[ii][kk].data.cpu().numpy().copy()
        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = []
            for ii in range(0, vid_emb.__len__()):
                video_embs_2 = []
                for kk in range(0, vid_emb[ii].__len__()):
                    video_embs_2.append(np.zeros((len(data_loader.dataset), vid_emb[0][0].size(1))))
                video_embs.append(video_embs_2)

            cap_embs = []
            for ii in range(0, cap_emb.__len__()):
                cap_embs_2 = []
                for kk in range(0, cap_emb[ii].__len__()):
                    cap_embs_2.append(np.zeros((len(data_loader.dataset), cap_emb[0][0].size(1))))
                cap_embs.append(cap_embs_2)

        # preserve the embeddings by copying from gpu and converting to numpy
        # video_embs[0][0][idxs] = vid_emb[0][0].data.cpu().numpy().copy()
        for ii in range(0, vid_emb.__len__()):
            for kk in range(0, vid_emb[ii].__len__()):
                video_embs[ii][kk][idxs] = vid_emb[ii][kk].data.cpu().numpy().copy()

        # cap_embs[0][idxs] = cap_emb[0].data.cpu().numpy().copy()
        # for ii in range(1, cap_emb.__len__()):
        #     cap_embs[ii][idxs] = cap_emb[ii].data.cpu().numpy().copy()
        for ii in range(0, cap_emb.__len__()):
            for kk in range(0, cap_emb[ii].__len__()):
                cap_embs[ii][kk][idxs] = cap_emb[ii][kk].data.cpu().numpy().copy()
        # video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        # cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs, video_ids, caption_ids, cap_embs_rand
    else:
        return video_embs, cap_embs, cap_embs_rand


def encode_data(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    video_ids = [''] * len(data_loader.dataset)
    caption_ids = [''] * len(data_loader.dataset)
    for i, (videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, *(cap_ids, vid_ids))

        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = []
            for ii in range(0, vid_emb.__len__()):
                video_embs_2 = []
                for kk in range(0, vid_emb[ii].__len__()):
                    video_embs_2.append(np.zeros((len(data_loader.dataset), vid_emb[0][0].size(1))))
                video_embs.append(video_embs_2)

            cap_embs = []
            for ii in range(0, cap_emb.__len__()):
                cap_embs_2 = []
                for kk in range(0, cap_emb[ii].__len__()):
                    cap_embs_2.append(np.zeros((len(data_loader.dataset), cap_emb[0][0].size(1))))
                cap_embs.append(cap_embs_2)

        # preserve the embeddings by copying from gpu and converting to numpy
        # video_embs[0][0][idxs] = vid_emb[0][0].data.cpu().numpy().copy()
        for ii in range(0, vid_emb.__len__()):
            for kk in range(0, vid_emb[ii].__len__()):
                video_embs[ii][kk][idxs] = vid_emb[ii][kk].data.cpu().numpy().copy()

        # cap_embs[0][idxs] = cap_emb[0].data.cpu().numpy().copy()
        # for ii in range(1, cap_emb.__len__()):
        #     cap_embs[ii][idxs] = cap_emb[ii].data.cpu().numpy().copy()
        for ii in range(0, cap_emb.__len__()):
            for kk in range(0, cap_emb[ii].__len__()):
                cap_embs[ii][kk][idxs] = cap_emb[ii][kk].data.cpu().numpy().copy()
        # video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        # cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs, video_ids, caption_ids
    else:
        return video_embs, cap_embs


def encode_data_1fea(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    video_ids = [''] * len(data_loader.dataset)
    caption_ids = [''] * len(data_loader.dataset)
    for i, (videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)

        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = []
            cap_embs = []
            video_embs.append(np.zeros((len(data_loader.dataset), vid_emb.size(1))))
            cap_embs.append(np.zeros((len(data_loader.dataset), cap_emb.size(1))))


        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[0][idxs] = vid_emb[0].data.cpu().numpy().copy()
        cap_embs[0][idxs] = cap_emb[0].data.cpu().numpy().copy()
        for ii in range(1, video_embs.__len__()):
            video_embs[ii][idxs] = vid_emb[ii].data.cpu().numpy().copy()
            cap_embs[ii][idxs] = cap_emb[ii].data.cpu().numpy().copy()

        # video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        # cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs, video_ids, caption_ids
    else:
        return video_embs, cap_embs


def encode_data_for_avs(model, data_loader, log_step=100, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    errorlists = []
    diagonal = []
    cap_ids_all = []
    video_ids = [''] * len(data_loader.dataset)
    caption_ids = [''] * len(data_loader.dataset)
    diagonal_ids = [''] * len(data_loader.dataset)

    for i, (videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_emb(videos, captions, True)

        # initialize the numpy arrays given the size of the embeddings
        # if video_embs is None:
        #    video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
        #     cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs = vid_emb.data.cpu().numpy().copy()
        cap_embs = cap_emb.data.cpu().numpy().copy()
        errorlistList = cosine_calculate(cap_embs, video_embs)

        errorlist = np.asanyarray(errorlistList)
        diagonal = np.append(diagonal, np.diag(errorlist))

        cap_ids_all.extend(cap_ids)
        # errorlists.extend(errorlist)

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]
            diagonal_ids[idx] = np.diag(errorlist).tolist()[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            # with open('data/errorlistTMP.data', 'wb') as filehandle:
            #     pickle.dump(diagonal, filehandle)
            #
            # with open('data/cap_ids_allTMP.data', 'wb') as filehandle:
            #     pickle.dump(cap_ids_all, filehandle)
            #
            # with open('data/diagonal_idsTMP.data', 'wb') as filehandle:
            #     pickle.dump(diagonal_ids, filehandle)
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs, diagonal, diagonal_ids, cap_ids_all, video_ids, caption_ids
    else:
        return video_embs, cap_embs, diagonal, diagonal_ids, cap_ids_all


# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i(c2i, vis_details=False, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == i // n_caption)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])


# recall@k, Med r, Mean r for Video-to-Text Retrieval
def i2t(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # remove duplicate videos
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds // n_caption == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2i_map(c2i, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0] * len(d_i)
        labels[i // n_caption] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def i2t_map(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]
        labels = [0] * len(d_i)
        labels[i * n_caption:(i + 1) * n_caption] = [1] * n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


def t2i_inv_rank(c2i, n_caption=1):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[0])

    for i in range(len(inv_ranks)):
        d_i = c2i[i, :]
        inds = np.argsort(d_i)

        rank = np.where(inds == i / n_caption)[0]
        inv_ranks[i] = sum(1.0 / (rank + 1))

    return np.mean(inv_ranks)


def i2t_inv_rank(c2i, n_caption=1):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    n_caption: number of captions of each image/video
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    for i in range(len(inv_ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds / n_caption == i)[0]
        inv_ranks[i] = sum(1.0 / (rank + 1))

    return np.mean(inv_ranks)


def i2t_inv_rank_multi(c2i, n_caption=2):
    """
    Text->videos (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    result = []
    for i in range(n_caption):
        idx = range(i, c2i.shape[0], n_caption)
        sub_c2i = c2i[idx, :]
        score = i2t_inv_rank(sub_c2i, n_caption=1)
        result.append(score)
    return result
