import torch
import torch.utils.data as data
import numpy as np
from basic.util import getVideoId
from util.vocab import clean_str
from pytorch_transformers import BertTokenizer
import itertools

import random
import clip
VIDEO_MAX_LEN = 64


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


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


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # print '\n******************************************'
    # print 'Video Process'
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids, tokens_tensor, segments_tensors, caption_text, CLIP_token = zip(*data)
    videos, captions, cap_bows, idxs, cap_ids, video_ids, tokens_tensor, segments_tensors, caption_text = zip(*data)

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
            # print(end)
            # print(max(video_lengths))
            if end < max(video_lengths):
                try:
                    # num_of_filler_frames = random.sample(range(0, end), max(video_lengths)-end)
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

    # print 'BERT Process'
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

    CLIP_token = torch.squeeze(clip.tokenize([a.decode("utf-8") for a in caption_text], truncate=True).clone().detach())
    video_data = (vidoes_all, videos_origin_all, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert, caption_text, CLIP_token)

    return video_data, text_data, idxs, cap_ids, video_ids


def collate_frame_gru_fn_frame_provider(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # print '\n******************************************'
    # print 'Video Process'

    videos, idxs, video_ids = zip(*data)

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
            # print(end)
            # print(max(video_lengths))
            if end < max(video_lengths):
                try:
                    # num_of_filler_frames = random.sample(range(0, end), max(video_lengths)-end)
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

    video_data = (vidoes_all, videos_origin_all, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_frame_gru_fn_text_provider(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # print '\n******************************************'
    # print 'Video Process'

    # videos, captions, cap_bows, idxs, cap_ids, video_ids, tokens_tensor, segments_tensors, caption_text, CLIP_token = zip(*data)
    captions, cap_bows, idxs, cap_ids, tokens_tensor, segments_tensors, caption_text = zip(*data)

    # print 'Text Process'
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

    # print 'BERT Process'
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

    # print '\n******************************************'
    # print words_mask.shape
    # print tokens_tensor_padded.shape
    # print segments_tensors_padded.shape
    # print len(lengths_bert)
    # CLIP_token = torch.stack(CLIP_token, 0) if CLIP_token[0] is not None else None
    # CLIP_token = torch.squeeze(torch.tensor(clip.tokenize([a.decode("utf-8") for a in caption_text], truncate=True)))
    CLIP_token = torch.squeeze(clip.tokenize([a.decode("utf-8") for a in caption_text], truncate=True).clone().detach())
    if CLIP_token.size().__len__() == 1:
        CLIP_token = torch.unsqueeze(CLIP_token, dim=0)
    text_data = (target, cap_bows, lengths, words_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert, caption_text, CLIP_token)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, bow2vec, vocab, do_visual_feas_norm, n_caption=None, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        self.do_visual_feas_norm = do_visual_feas_norm
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.CLIP_model, self.CLIP_preprocess = clip.load("ViT-B/32")

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                # print(line)
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
        if n_caption is not None:
            print(n_caption)
            assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]

        frames_tensors = []
        for vis_fea in self.visual_feat:
            frame_vecs = []
            for frame_id in frame_list:
                # l_2 normalize
                if (self.do_visual_feas_norm):
                    frame_vecs.append(do_L2_norm(self.visual_feat[vis_fea].read_one(frame_id)))
                else:
                    frame_vecs.append(self.visual_feat[vis_fea].read_one(frame_id))
            frames_tensor = torch.Tensor(np.array(frame_vecs))
            frames_tensors.append(frames_tensor)
        # text
        # print video_id
        cap_text = self.captions[cap_id]
        caption_text = cap_text[:]
        caption_text = ' '.join(clean_str(caption_text))
        caption_text = text2Berttext(caption_text, self.tokenizer)
        caption_text = caption_text.encode("utf-8").decode("utf-8")

        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption_text)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption_text)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        # BERT
        caption_text = cap_text[:]
        caption_text = ' '.join(clean_str(caption_text))
        marked_text = "[CLS] " + caption_text + " [SEP]"
        # print marked_text
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # tmptext = self.tokenizer.convert_tokens_to_string(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        # caption_text = cap_text[:]
        # caption_text = text2Berttext(caption_text, self.tokenizer)
        caption_text = caption_text.encode("utf-8")

        # CLIP encoding
        # CLIP_token = torch.squeeze( torch.tensor( clip.tokenize([caption_text.decode("utf-8")], truncate=True) ) )
        # print tokens_tensor.shape
        # print caption.__len__()
        # return frames_tensors, cap_tensor, cap_bow, index, cap_id, video_id, tokens_tensor, segments_tensors, caption_text, CLIP_token
        return frames_tensors, cap_tensor, cap_bow, index, cap_id, video_id, tokens_tensor, segments_tensors, caption_text

    def __len__(self):
        return self.length


class Dataset4DualEncoding_frame_provider(data.Dataset):
    """
       Load captions and video frame features by pre-trained CNN model.
       """
    def __init__(self,  visual_feat, do_visual_feas_norm, video2frames=None):

        # self.video_ids = set()
        self.video2frames = video2frames
        self.do_visual_feas_norm = do_visual_feas_norm

        self.video_ids = [key for key in self.video2frames.keys()]
        self.visual_feat = visual_feat

        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # video
        frame_list = self.video2frames[video_id]

        frames_tensors = []
        for vis_fea in self.visual_feat:
            frame_vecs = []
            for frame_id in frame_list:
                # l_2 normalize
                if (self.do_visual_feas_norm):
                    frame_vecs.append(do_L2_norm(self.visual_feat[vis_fea].read_one(frame_id)))
                else:
                    frame_vecs.append(self.visual_feat[vis_fea].read_one(frame_id))
            frames_tensor = torch.Tensor(np.array(frame_vecs))
            frames_tensors.append(frames_tensor)

        return frames_tensors, index, video_id

    def __len__(self):
        return self.length


class Dataset4DualEncoding_text_provider(data.Dataset):
    """
    Load captions.
    """

    def __init__(self, cap_file,  bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.CLIP_model, self.CLIP_preprocess = clip.load("ViT-B/32")

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                # print(line)
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        # text
        # print video_id
        cap_text = self.captions[cap_id]
        caption_text = cap_text[:]
        caption_text = ' '.join(clean_str(caption_text))
        caption_text = text2Berttext(caption_text, self.tokenizer)
        caption_text = caption_text.encode("utf-8").decode("utf-8")

        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption_text)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption_text)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        # BERT
        caption_text = cap_text[:]
        caption_text = ' '.join(clean_str(caption_text))
        marked_text = "[CLS] " + caption_text + " [SEP]"
        # print marked_text
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # tmptext = self.tokenizer.convert_tokens_to_string(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        # caption_text = cap_text[:]
        # caption_text = text2Berttext(caption_text, self.tokenizer)
        caption_text = caption_text.encode("utf-8")

        return cap_tensor, cap_bow, index, cap_id, tokens_tensor, segments_tensors, caption_text

    def __len__(self):
        return self.length


def get_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=5, n_caption=2, do_visual_feas_norm=1, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, do_visual_feas_norm,
                                          video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, do_visual_feas_norm, n_caption,
                                        video2frames=video2frames['val'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=(x == 'train'),
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn,
                                                   drop_last=(x == 'train'))
                    for x in cap_files}
    return data_loaders


def get_test_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2, do_visual_feas_norm=1,
                          video2frames=None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, do_visual_feas_norm, n_caption,
                                         video2frames=video2frames['test'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn)
                    for x in cap_files}
    return data_loaders


def get_data_loaders_frame_provider(visual_feats, batch_size=100, num_workers=5, do_visual_feas_norm=1, video2frames=None):
    dset = Dataset4DualEncoding_frame_provider(visual_feats['test'], do_visual_feas_norm, video2frames=video2frames)
    data_loaders = torch.utils.data.DataLoader(dataset=dset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn_frame_provider)

    return data_loaders


def get_data_loaders_text_provider(cap_file, vocab, bow2vec, batch_size=100, num_workers=5):
    dset = Dataset4DualEncoding_text_provider(cap_file, bow2vec, vocab)
    data_loaders = torch.utils.data.DataLoader(dataset=dset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn_text_provider)

    return data_loaders


if __name__ == '__main__':
    pass
