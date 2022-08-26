import torch
import torch.utils.data as data
import numpy as np
from basic.bigfile import BigFile


import itertools
import random
VIDEO_MAX_LEN = 64


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
    return 1.0 * np.array(vec) / L2_norm


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # print '\n******************************************'
    # print 'Video Process'
    # Sort a data list by caption length
    # if data[0][1] is not None:
    #     data.sort(key=lambda x: len(x[1]), reverse=True)
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


class Dataset4DualEncoding(data.Dataset):
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


def get_test_data_loaders(visual_feats, batch_size=100, num_workers=5, do_visual_feas_norm=1, video2frames=None):
    dset = Dataset4DualEncoding(visual_feats, do_visual_feas_norm, video2frames=video2frames)
    data_loaders = torch.utils.data.DataLoader(dataset=dset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn)

    return data_loaders