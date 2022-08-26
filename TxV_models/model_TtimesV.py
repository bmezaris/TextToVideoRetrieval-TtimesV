import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from loss import cosine_sim, MarginRankingLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from torch.autograd import Variable
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np

import torch.nn.functional as F

from basic.bigfile import BigFile
import clip

# from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we))
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm + 1e-10)
    return X


def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class BertEmbedding(nn.Module):

    def __init__(self, opt):
        super(BertEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load pre-trained model (weights)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bert_model.eval()

    def forward(self, tokens_tensor, segments_tensors):
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers = self.bert_model(tokens_tensor, segments_tensors)

        last_hidden_state = encoded_layers[0]  # The last hidden-state is the first element of the output tuple
        pooler_output = encoded_layers[1]
        hidden_states = encoded_layers[2]

        return last_hidden_state


class TransformNet(nn.Module):
    def __init__(self, opt, fc_input_size, fc_output_size=None):
        super().__init__()

        self.fc1 = nn.Linear(fc_input_size, opt.text_mapping_layers[1])
        if opt.have_bn:
            self.bn1 = nn.BatchNorm1d(opt.text_mapping_layers[1])
        else:
            self.bn1 = None

        if opt.activation == 'tanh':
            self.activation = nn.Tanh()
        elif opt.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

        if opt.dropout > 1e-3:
            self.dropout = nn.Dropout(p=opt.dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.apply(_initialize_weights)

    def forward(self, input_x):
        features = self.fc1(input_x.to(device))

        if self.bn1 is not None:
            features = self.bn1(features)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        return features

    def load_state_dict(self, state_dict):
        """Copy parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super().load_state_dict(new_state)


class AttentionLayer(nn.Module):
    """
        Attention Layer
    """

    def __init__(self, fc_input, hidden_size):
        super(AttentionLayer, self).__init__()

        self.input_size = fc_input
        self.hidden_size = hidden_size
        self.output_size = fc_input

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc1.bias.data.fill_(0)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.fc2.bias.data.fill_(0)
        # self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """

        xavier_init_fc(self.fc1)
        xavier_init_fc(self.fc2)

    def forward(self, inputs):
        W_s1 = self.fc1(inputs)
        tanh = self.tanh(W_s1)
        W_s2 = self.fc2(tanh)
        # output = self.softmax(W_s2)
        output = W_s2
        return output


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.selected_text_feas = opt.selected_text_feas

        # visual bidirectional rnn encoder
        self.embed_w2v = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.embed_bert = BertEmbedding(opt)
        self.rnn = nn.GRU(opt.word_dim_concat, opt.text_rnn_size, batch_first=True, bidirectional=True)
        self.CLIP_model, self.CLIP_preprocess = clip.load("ViT-B/32")

        # Attention
        self.atten = AttentionLayer(self.rnn_output_size, self.rnn_output_size)
        # self.Selfatten = SelfAttention(self.rnn_output_size)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])

        self.Linears = nn.ModuleList([
            nn.ModuleList(
                [TransformNet(opt, self.select_text_input(tt, opt)) for vv in range(opt.visual_feat_dim.__len__())]) for
            tt in self.selected_text_feas
        ])

        self.init_weights()

    def select_text_input(self, type, opt):
        assert type in ['bow', 'w2v', 'bert', 'w2v-bert', 'gru', 'conv', 'att', 'clip'], '%s not supported.' % type

        if type == 'bow':
            size = opt.bow_vocab_size
        elif type == 'w2v':
            size = opt.word_dim
        elif type == 'bert':
            size = opt.bert_dim
        elif type == 'w2v-bert':
            size = opt.word_dim_concat
        elif type == 'gru':
            size = self.rnn_output_size
        elif type == 'conv':
            size = 2 + opt.text_kernel_num * len(opt.text_kernel_sizes)
        elif type == 'att':
            size = opt.text_mapping_layers[0]
        elif type == 'clip':
            size = opt.clip_mapping_layers

        return size

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed_w2v.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed_w2v.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors

        cap_wids, cap_bows, lengths, cap_mask, tokens_tensor_padded, segments_tensors_padded, lengths_bert, caption_text, CLIP_tensor = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids_w2v = self.embed_w2v(cap_wids)
        cap_wids_bert = self.embed_bert(tokens_tensor_padded, segments_tensors_padded)
        cap_wids_concat = torch.cat([cap_wids_w2v, cap_wids_bert], dim=2)

        packed = pack_padded_sequence(cap_wids_concat, lengths, batch_first=True)
        # tmp = (Variable(packed).data).cpu().numpy()
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]

        # Level 2B. Attention
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        # H_new = Variable(torch.zeros(gru_init_out.size(0), gru_init_out.size(1), self.rnn_output_size)).cuda()
        H_new_tmp_2 = self.atten(gru_init_out)
        # H_new_tmp_2 = self.Selfatten(gru_init_out)
        H_new_2 = gru_init_out * H_new_tmp_2

        for i, batch in enumerate(H_new_2):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 2B. Attention
        # gru_out = self.atten(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        # con_out = gru_init_out.unsqueeze(1)
        con_out = H_new_2.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        with torch.no_grad():
            CLIP_features = self.CLIP_model.encode_text(CLIP_tensor).float()

        # concatenation
        if self.concate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)

        # mapping to common space
        feas = []
        for jj, ff in enumerate(self.Linears):
            type = self.selected_text_feas[jj]

            if type == 'bow':
                # features_tmp = self.Linears[jj](org_out)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](org_out)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'w2v':
                tmp_mean = Variable(torch.zeros(cap_wids_w2v.size()[0], cap_wids_w2v.size()[-1])).cuda()
                for i, batch in enumerate(cap_wids_w2v):
                    tmp_mean[i] = torch.mean(batch[:lengths[i]], 0)
                # features_tmp = self.Linears[jj](tmp_mean)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](tmp_mean)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'bert':
                tmp_mean = Variable(torch.zeros(cap_wids_bert.size()[0], cap_wids_bert.size()[-1])).cuda()
                for i, batch in enumerate(cap_wids_bert):
                    tmp_mean[i] = torch.mean(batch[:lengths[i]], 0)

                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](tmp_mean)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'w2v-bert':
                tmp_mean = Variable(torch.zeros(cap_wids_concat.size()[0], cap_wids_concat.size()[-1])).cuda()
                for i, batch in enumerate(cap_wids_concat):
                    tmp_mean[i] = torch.mean(batch[:lengths[i]], 0)
                # features_tmp = self.Linears[jj](tmp_mean)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](tmp_mean)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'gru':
                # features_tmp = self.Linears[jj](gru_out)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](gru_out)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'conv':
                # features_tmp = self.Linears[jj](con_out)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](con_out)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'att':
                # features_tmp = self.Linears[jj](features)
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](features)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)

            elif type == 'clip':
                feas_tmp = []
                for kk, cc in enumerate(self.Linears[jj]):
                    features_tmp = self.Linears[jj][kk](CLIP_features)
                    if self.text_norm:
                        feas_tmp.append(l2norm(features_tmp))
                    else:
                        feas_tmp.append(features_tmp)
                feas.append(feas_tmp)
        return feas


class Video_multilevel_encoding(nn.Module):

    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()
        self.num_of_spaces = opt.txt_net_list
        self.visual_norm = opt.visual_norm

        self.FC_1 = TransformNet(opt, opt.visual_feat_dim[0])
        self.visfea_1 = nn.ModuleList([TransformNet(opt, opt.visual_feat_dim[0]) for tt in opt.selected_text_feas])

        self.FC_2 = TransformNet(opt, opt.visual_feat_dim[1])
        self.visfea_2 = nn.ModuleList([TransformNet(opt, opt.visual_feat_dim[1]) for tt in opt.selected_text_feas])

        self.FC_3 = TransformNet(opt, opt.visual_feat_dim[2])
        self.visfea_3 = nn.ModuleList([TransformNet(opt, self.FC_3.fc1.out_features) for tt in opt.selected_text_feas])

    def forward(self, videos):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos

        # fea_1_gcn = self.GCN_1(videos[0])
        tmp_mean_1 = Variable(torch.zeros(videos[0].size()[0], videos[0].size()[-1])).cuda()
        for i, batch in enumerate(videos[0]):
            tmp_mean_1[i] = torch.mean(batch[:lengths[i]], 0)
        fea_FC_1 = self.FC_1(tmp_mean_1)

        feas_1 = []
        for jj, ff in enumerate(self.visfea_1):
            fea = self.visfea_1[jj](fea_FC_1)
            if self.visual_norm:
                feas_1.append(l2norm(fea))
            else:
                feas_1.append(fea)

        tmp_mean_2 = Variable(torch.zeros(videos[1].size()[0], videos[1].size()[-1])).cuda()
        for i, batch in enumerate(videos[1]):
            tmp_mean_2[i] = torch.mean(batch[:lengths[i]], 0)
        fea_FC_2 = self.FC_2(tmp_mean_2)

        feas_2 = []
        for jj, ff in enumerate(self.visfea_2):
            fea = self.visfea_2[jj](fea_FC_2)
            if self.visual_norm:
                feas_2.append(l2norm(fea))
            else:
                feas_2.append(fea)

        tmp_mean_3 = Variable(torch.zeros(videos[2].size()[0], videos[2].size()[-1])).cuda()
        for i, batch in enumerate(videos[2]):
            tmp_mean_3[i] = torch.mean(batch[:lengths[i]], 0)
        fea_FC_3 = self.FC_3(tmp_mean_3)

        feas_3 = []
        for jj, ff in enumerate(self.visfea_3):
            fea = self.visfea_3[jj](fea_FC_3)
            if self.visual_norm:
                feas_3.append(l2norm(fea))
            else:
                feas_3.append(fea)

        return feas_1, feas_2, feas_3


class CrossModalNet(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()

    def _init_optim(self, opt):
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)]

        self.Eiters = 0

    def _init_loss(self, opt):
        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           similarity=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction)

    def lr_step(self, val_value):
        self.lr_schedulers[0].step()
        self.lr_schedulers[1].step(val_value)

    def train_emb(self, videos, captions, lengths, *args):
        """One training step given vis_feats and captions.
                """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_emb(videos, captions, False)
        # txt_embs = self.text_encoding(captions)
        # vis_embs = self.vid_encoding(videos)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, indices_im = self.compute_loss_multiple(cap_emb, vid_emb)

        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return loss_value, vid_emb[0][0].size(0)


class TtimesV(CrossModalNet):

    def __init__(self, opt):
        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)

        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        self.params = list(self.text_encoding.parameters())
        self.params += list(self.vid_encoding.parameters())

        self._init_loss(opt)
        self._init_optim(opt)

        self.similarity = opt.measure
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)]

        self.Eiters = 0

    def forward_emb(self, videos, targets, *args):
        """Compute the video and caption embeddings
               """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos

        feas_1, feas_2, feas_3 = frames
        feas_1 = Variable(feas_1)
        feas_2 = Variable(feas_2)
        feas_3 = Variable(feas_3)
        if torch.cuda.is_available():
            feas_1 = feas_1.cuda()
            feas_2 = feas_2.cuda()
            feas_3 = feas_3.cuda()
        frames = (feas_1, feas_2, feas_3)

        feas_1, feas_2, feas_3 = mean_origin
        feas_1 = Variable(feas_1)
        feas_2 = Variable(feas_2)
        feas_3 = Variable(feas_3)
        if torch.cuda.is_available():
            feas_1 = feas_1.cuda()
            feas_2 = feas_2.cuda()
            feas_3 = feas_3.cuda()
        mean_origin = (feas_1, feas_2, feas_3)

        vidoes_mask = Variable(vidoes_mask)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

        # text data
        captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert, caption_text, CLIP_token = targets
        if captions is not None:
            captions = Variable(captions)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()

        if tokens_tensor_padded is not None:
            tokens_tensor_padded = Variable(tokens_tensor_padded)
            if torch.cuda.is_available():
                tokens_tensor_padded = tokens_tensor_padded.cuda()

        if segments_tensors_padded is not None:
            segments_tensors_padded = Variable(segments_tensors_padded)
            if torch.cuda.is_available():
                segments_tensors_padded = segments_tensors_padded.cuda()

        if CLIP_token is not None:
            CLIP_token = Variable(CLIP_token)
            if torch.cuda.is_available():
                CLIP_token = CLIP_token.cuda()

        text_data = (
            captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert,
            caption_text, CLIP_token)

        vid_emb = self.vid_encoding(videos_data)
        cap_emb = self.text_encoding(text_data)
        return vid_emb, cap_emb

    def compute_loss_multiple(self, txt_embs, vis_embs):
        """Compute the loss given pairs of image and caption embeddings
        """

        loss = 0
        indices_im = None
        for i in range(0, len(vis_embs)):
            for j in range(len(txt_embs)):
                # a = txt_embs[j][i]
                # b = vis_embs[i][j]
                cur_loss = self.criterion(txt_embs[j][i], vis_embs[i][j])
                loss += cur_loss
        return loss, indices_im

    def forward_vis(self, video_data, idxs, video_ids):
        """Compute the video embeddings
                       """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = video_data

        feas_1, feas_2, feas_3 = frames
        feas_1 = Variable(feas_1)
        feas_2 = Variable(feas_2)
        feas_3 = Variable(feas_3)
        if torch.cuda.is_available():
            feas_1 = feas_1.cuda()
            feas_2 = feas_2.cuda()
            feas_3 = feas_3.cuda()
        frames = (feas_1, feas_2, feas_3)

        feas_1, feas_2, feas_3 = mean_origin
        feas_1 = Variable(feas_1)
        feas_2 = Variable(feas_2)
        feas_3 = Variable(feas_3)
        if torch.cuda.is_available():
            feas_1 = feas_1.cuda()
            feas_2 = feas_2.cuda()
            feas_3 = feas_3.cuda()
        mean_origin = (feas_1, feas_2, feas_3)

        vidoes_mask = Variable(vidoes_mask)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

        vid_emb = self.vid_encoding(videos_data)

        return vid_emb

    def forward_text(self, text_data, *args):
        """Compute the video and caption embeddings
               """

        # text data
        captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert, caption_text, CLIP_token = text_data
        if captions is not None:
            captions = Variable(captions)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()

        if tokens_tensor_padded is not None:
            tokens_tensor_padded = Variable(tokens_tensor_padded)
            if torch.cuda.is_available():
                tokens_tensor_padded = tokens_tensor_padded.cuda()

        if segments_tensors_padded is not None:
            segments_tensors_padded = Variable(segments_tensors_padded)
            if torch.cuda.is_available():
                segments_tensors_padded = segments_tensors_padded.cuda()

        if CLIP_token is not None:
            CLIP_token = Variable(CLIP_token)
            if torch.cuda.is_available():
                CLIP_token = CLIP_token.cuda()

        text_data = (
            captions, cap_bows, lengths, cap_masks, tokens_tensor_padded, segments_tensors_padded, lengths_bert,
            caption_text, CLIP_token)

        cap_emb = self.text_encoding(text_data)
        return cap_emb


NAME_TO_MODEL = {
    'TtimesV': TtimesV
}


def get_model(name):
    assert name in NAME_TO_MODEL, '%s not supported.' % name
    return NAME_TO_MODEL[name]


if __name__ == '__main__':
    model = get_model('TtimesV')
