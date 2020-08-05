import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import Counter
import src.utils as utils


EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

debug_loss = []
debug_loss_grad = None
last_weight = None
last_log_prob = None
last_weight_grad = None

action_dist_debug1 = None
action_dist_debug2 = None
action_dist_debug3 = None

class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class CsgoModel(ReptileModel):
    def __init__(self, args, npy_dict):
        super(CsgoModel, self).__init__()
        self.args = args
        self.embedding = torch.tensor(npy_dict["action_embedding"]).cuda().float()
        self.id2name = list(npy_dict["action_name"])
        self.id2money = list(npy_dict["action_money"])
        self.prices = torch.tensor(self.id2money).float().cuda()
        self.action_capacity = list(npy_dict["action_capacity"])
        self.type_capacity = list(npy_dict["type_capacity"])
        self.id2type = npy_dict["id2type"]
        self.typeid2name = list(npy_dict["typeid2name"])

        self.embedding_dim = self.embedding.size()[1]
        self.output_dim = self.embedding.size()[0]

        self.end_idx = self.embedding.size()[0] - 1
        self.start_idx = self.end_idx
        self.action_capacity[self.end_idx] = HUGE_INT
        self.type_capacity[self.id2type[self.end_idx]] = HUGE_INT

        self.mute_action_mask = 1.0 - torch.eye(len(self.id2type)).cuda()
        self.mute_type_mask = torch.ones(len(self.typeid2name), len(self.id2type)).float().cuda()
        for i, typeid in enumerate(self.id2type):
            self.mute_type_mask[typeid][i] = 0.0
        self.output_categories = 3


        self.category_offset = [0, int((self.id2type<=5).sum()), int((self.id2type<=6).sum())]
        self.category_action_offset = []  # lstm dist id -> action id
        self.output_dim1 = self.category_offset[1] + 1
        self.output_dim2 = self.category_offset[2] - self.category_offset[1] + 1
        self.output_dim3 = self.output_dim - self.category_offset[2]
        self.output_dims = [self.output_dim1, self.output_dim2, self.output_dim3]
        assert self.output_dim == self.output_dim1 + self.output_dim2 + self.output_dim3 - 2
        self.category_action_offset.append(list(np.arange(self.output_dim1 - 1)) + [self.end_idx])
        self.category_action_offset.append(list(np.arange(self.output_dim2 - 1) + self.category_offset[1]) + [self.end_idx])
        self.category_action_offset.append(list(np.arange(self.output_dim3) + self.category_offset[2]))

        self.money_scaling = args.money_scaling

        self.shared_attention_weight = args.shared_attention_weight
        self.different_attention_weight = args.different_attention_weight
        # ways of combining history representation of self
        self.hist_encoding = args.history_encoding
        self.time_decaying = args.time_decaying
        self.lstm_mode = args.lstm_mode

        side_mask = npy_dict["side_mask"]
        self.side_mask = []
        self.side_mask.append(torch.tensor(side_mask['t_mask'].astype(float)).cuda().unsqueeze(0))
        self.side_mask.append(torch.tensor(side_mask['ct_mask'].astype(float)).cuda().unsqueeze(0))
        self.side_mask = torch.cat(self.side_mask, 0)

        self.side_embedding = torch.tensor([[1, 0], [0, 1]]).cuda().float()
        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.ff_dim = args.ff_dim
        self.resource_dim = args.resource_dim
        self.money_dim = 5 # only allies
        self.team_dim = 2
        if self.hist_encoding is not None:
            self.input_dim = self.embedding_dim * 4 + self.money_dim + self.team_dim
        else:
            self.input_dim = self.embedding_dim * 3 + self.money_dim + self.team_dim

        self.ff_dropout_rate = args.ff_dropout_rate
        self.max_output_num = args.max_output_num
        self.beam_size = args.beam_size

        self.define_modules()
        #xavier_initialization
        self.initialize_modules()

    def reward_fun(self, a, a_r):
        # F1 score
        a_new = a
        a_r_new = a_r
        # both are empty
        if len(a_new) == 0 and len(a_r_new) == 0:
            return 1.0
        a_common = list((Counter(a_new) & Counter(a_r_new)).elements())

        # weighting using prices
        '''tp = torch.sum(self.prices[a_common])
        recall = tp / (torch.sum(self.prices[a_r_new]) + EPSILON)
        precision = tp / (torch.sum(self.prices[a_new]) + EPSILON)'''

        # unweighted
        recall = len(a_common) / (len(a_r_new) + EPSILON)
        precision = len(a_common) / (len(a_new) + EPSILON)

        F1_score = 2 * precision * recall / (precision + recall + EPSILON)
        return F1_score

    def loss(self, predictions, labels):
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        action_list, greedy_list, action_prob_by_category, no_output_bi, bi_prob, action_list_by_category, greedy_list_by_category = predictions


        batch_size = len(action_list)

        no_output_bi = torch.tensor(no_output_bi).cuda()

        log_bi_probs = torch.log(bi_prob)

        bi_labels = torch.tensor(utils.get_batched_category_label(labels, self.id2type)).cuda().float()

        # LSTM loss
        seq_loss_print = []
        seq_loss = torch.tensor(0.0).cuda()

        if self.lstm_mode == 'triple':
            label_by_batch = utils.filter_batched_category_actions(labels, self.id2type, self.output_categories)
            label_by_category = utils.reshape_batched_category_actions(label_by_batch)
            
            for i in range(self.output_categories):
                loss_cat = []
                if torch.sum(no_output_bi[:, i]) == batch_size:
                    seq_loss_print.append(np.nan)
                else:
                    for j in range(batch_size):
                        reward_sample = self.reward_fun(action_list_by_category[i][j], label_by_category[i][j])
                        reward_greedy = self.reward_fun(greedy_list_by_category[i][j], label_by_category[i][j])
                        reward = reward_sample - reward_greedy
                        log_prob = torch.sum(torch.log(action_prob_by_category[i][j]))
                        seq_loss_category = -reward * log_prob
                        seq_loss += seq_loss_category
                        if no_output_bi[j][i]:
                            loss_cat.append(np.nan)
                        else:
                            loss_cat.append(seq_loss_category.detach().item())
                    seq_loss_print.append(np.nanmean(loss_cat))            
        elif self.lstm_mode == 'single':
            loss_cat = []
            if torch.sum(no_output_bi) == batch_size:
                seq_loss_print.append(np.nan)
            else:
                for i in range(batch_size):
                    reward_sample = self.reward_fun(action_list[i], labels[i])
                    reward_greedy = self.reward_fun(greedy_list[i], labels[i])
                    reward = reward_sample - reward_greedy
                    log_prob = torch.sum(torch.log(action_prob_by_category[0][i]))
                    seq_loss_category = -reward * log_prob
                    seq_loss += seq_loss_category
                    if no_output_bi[i]:
                        loss_cat.append(np.nan)
                    else:
                        loss_cat.append(seq_loss_category.detach().item())
                seq_loss_print.append(np.nanmean(loss_cat))
                
        seq_loss /= batch_size

        # binary classifier loss
        bi_loss = -torch.sum(log_bi_probs * bi_labels) / batch_size
        bi_loss_print = list((torch.sum(log_bi_probs * bi_labels, 0) / batch_size).cpu().detach().numpy())


        loss_dict = {}
        loss_dict['model_loss'] = seq_loss.double() + bi_loss.double()
        loss_dict['bi_loss'] = bi_loss_print
        loss_dict['seq_loss'] = seq_loss_print
        
        loss_dict['real_seq_loss'] = seq_loss.double()

        return loss_dict

    def get_embedding(self, idx):
        return self.embedding[idx]

    def high_att(self, x, side = None):
        # team-level attention
        if side is None:
            h = self.att_LN2(x)
        elif side == 't':
            h = self.att_LN2_t(x)
        elif side == 'o':
            h = self.att_LN2_o(x)
        h = torch.tanh(h)
        if side is None:
            h = self.v2(h)
        elif side == 't':
            h = self.v2_t(h)
        elif side == 'o':
            h = self.v2_o(h)
        att = F.softmax(h, 0)
        ret = torch.sum(att * x, 0)
        return ret

    def low_att(self, x, side = None):
        # player-level attention
        if side is None:
            h = self.att_LN1(x)
        elif side == 's':
            h = self.att_LN1_s(x)
        elif side == 't':
            h = self.att_LN1_t(x)
        elif side == 'o':
            h = self.att_LN1_o(x)
        elif side == 'h':
            h = self.att_LN1_h(x)
        h = torch.tanh(h)
        if side is None:
            h = self.v1(h)
        elif side == 's':
            h = self.v1_s(h)
        elif side == 't':
            h = self.v1_t(h)
        elif side == 'o':
            h = self.v1_o(h)
        elif side == 'h':
            h = self.v1_h(h)
        att = F.softmax(h, 0)
        ret = torch.sum(att * x, 0)
        return ret

    def encode_money(self, x):
        h = self.money_LN1(x)
        h = F.relu(h)
        h = self.LNDropout(h)
        h = self.money_LN2(h)
        h = F.relu(h)
        return h

    def BiClassif(self, x, category_id=None):
        if category_id is None:
            h_bi = self.BClassif1(x)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif2(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif3(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif4(h_bi)
        elif category_id == 0:
            h_bi = self.BClassif1_1(x)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif1_2(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif1_3(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif1_4(h_bi)
        elif category_id == 1:
            h_bi = self.BClassif2_1(x)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif2_2(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif2_3(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif2_4(h_bi)
        elif category_id == 2:
            h_bi = self.BClassif3_1(x)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif3_2(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif3_3(h_bi)
            h_bi = F.relu(h_bi)
            h_bi = self.LNDropout(h_bi)
            h_bi = self.BClassif3_4(h_bi)
        else:
            raise NotImplementedError("Category ID exceeds number of output categories.")

        return F.softmax(h_bi, dim = -1)

    def classif_LN(self, x, category_id=None):
        if category_id is None:
            out = self.LN1(x)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN2(out)
        elif category_id == 0:
            out = self.LN1_1(x)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN1_2(out)
        elif category_id == 1:
            out = self.LN2_1(x)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN2_2(out)
        elif category_id == 2:
            out = self.LN3_1(x)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN3_2(out)
        else:
            raise NotImplementedError("Category ID exceeds number of output categories.")

        out = F.softmax(out, dim = -1)
        return out

    def money_mask(self, money):
        return (self.prices <= money).float().cuda()

    def get_residual_capacity(self, l, capacity):
        possession = Counter(l)
        residual_capacity = capacity.copy()
        for key, value in possession.items():
            residual_capacity[key] -= value
        return residual_capacity

    def get_capacity_mask(self, res_capacity, mute_mask):
        to_multiply_mask = 1 - (res_capacity > 0).float()
        ret = (torch.mm(to_multiply_mask, mute_mask) == torch.sum(to_multiply_mask, 1).unsqueeze(1)).float()
        return ret
    
    def generate(self, init_embedding, init_h, init_c, no_output_bi, money, side_all, res_action_capacity, res_type_capacity, is_greedy=True):
        '''
        generate predictions using lstm
        '''
        action_list = []
        action_list_by_category = []
        action_prob_by_category = []
        batch_size = len(init_embedding)
        if self.lstm_mode == 'triple':
            for j in range(self.output_categories):
                # initialize LSTM
                self.initialize_lstm(init_embedding, (init_h, init_c), j)

                action_prob_category = []
                action_list_category = []
                is_end = no_output_bi[:, j].float()

                for i in range(self.max_output_num):
                    if torch.sum(is_end) == len(is_end):
                        for _ in range(self.max_output_num - i):
                            action_list.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                            action_list_category.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                            action_prob_category.append(torch.ones(batch_size, 1).double().cuda())
                        break

                    H = self.history[-1][0][-1, :, :]
                    action_dist = self.classif_LN(H, j)

                    action_mask = self.money_mask(money.unsqueeze(1)) * self.side_mask[side_all]
                    action_mask *= (res_action_capacity > 0).float()
                    action_mask *= self.get_capacity_mask(res_type_capacity, self.mute_type_mask)
                    action_mask = action_mask[:, self.category_action_offset[j]]
                    action_dist = action_dist * action_mask # + (1 - action_mask) * EPSILON

                    is_zero = (torch.sum(action_dist, 1) == 0).float().unsqueeze(1)
                    mask_all_tensor = torch.zeros(action_dist.size()[1], dtype = torch.float).cuda()
                    mask_all_tensor[-1] = 1.0
                    action_dist = action_dist + is_zero * mask_all_tensor

                    if is_end is not None:
                        action_dist = (1 - is_end.unsqueeze(1)) * action_dist + is_end.unsqueeze(1) * mask_all_tensor

                    if is_greedy:
                        output_idx = torch.argmax(action_dist, 1).unsqueeze(1)
                    else:
                        # normalize
                        action_dist = action_dist / torch.sum(action_dist, 1).unsqueeze(1)
                        output_idx = torch.multinomial(action_dist, 1, replacement=True)

                    action_idx = torch.tensor(self.category_action_offset[j]).cuda()[output_idx]

                    action_prob_category.append(torch.gather(action_dist, 1, output_idx))

                    is_end = (action_idx == self.end_idx).view(-1).float()

                    action_list.append(action_idx)
                    action_list_category.append(action_idx)

                    self.update_lstm(action_idx, j)

                    money = money - self.prices[action_idx.view(-1)]
                    res_action_capacity -= torch.eye(res_action_capacity.size()[1]).cuda()[action_idx.view(-1)]
                    res_type_capacity -= torch.eye(res_type_capacity.size()[1]).cuda()[torch.tensor(self.id2type).cuda()[action_idx.view(-1)]]

                    assert torch.sum((money >= 0).float()) == batch_size
                    assert torch.sum((res_action_capacity >= 0).float()) == len(res_action_capacity.view(-1))
                    assert torch.sum((res_type_capacity >= 0).float()) == len(res_type_capacity.view(-1))

                action_list_by_category.append(utils.remove_token(torch.cat(action_list_category, 1).cpu().numpy().tolist(), self.end_idx))
                action_prob_by_category.append(torch.cat(action_prob_category, 1))
        elif self.lstm_mode == 'single':
            # initialize LSTM
            self.initialize_lstm(init_embedding, (init_h, init_c))

            # in single lstm mode, action_list_category is the same as action_list
            action_prob_category = []
            action_list_category = []
            is_end = no_output_bi.float().view(-1)

            for i in range(self.max_output_num):
                if torch.sum(is_end) == len(is_end):
                    for _ in range(self.max_output_num - i):
                        action_list.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                        action_list_category.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                        action_prob_category.append(torch.ones(batch_size, 1).double().cuda())
                    break

                H = self.history[-1][0][-1, :, :]
                action_dist = self.classif_LN(H)

                action_mask = self.money_mask(money.unsqueeze(1)) * self.side_mask[side_all]
                action_mask *= (res_action_capacity > 0).float()
                action_mask *= self.get_capacity_mask(res_type_capacity, self.mute_type_mask)
                action_dist = action_dist * action_mask # + (1 - action_mask) * EPSILON

                is_zero = (torch.sum(action_dist, 1) == 0).float().unsqueeze(1)
                mask_all_tensor = torch.zeros(action_dist.size()[1], dtype = torch.float).cuda()
                mask_all_tensor[-1] = 1.0
                action_dist = action_dist + is_zero * mask_all_tensor

                if is_end is not None:
                    action_dist = (1 - is_end.unsqueeze(1)) * action_dist + is_end.unsqueeze(1) * mask_all_tensor

                if is_greedy:
                    output_idx = torch.argmax(action_dist, 1).unsqueeze(1)
                else:
                    # normalize
                    action_dist = action_dist / torch.sum(action_dist, 1).unsqueeze(1)
                    output_idx = torch.multinomial(action_dist, 1, replacement=True)

                action_idx = output_idx

                action_prob_category.append(torch.gather(action_dist, 1, output_idx))

                is_end = (action_idx == self.end_idx).view(-1).float()

                action_list.append(action_idx)
                action_list_category.append(action_idx)

                self.update_lstm(action_idx)

                money = money - self.prices[action_idx.view(-1)]
                res_action_capacity -= torch.eye(res_action_capacity.size()[1]).cuda()[action_idx.view(-1)]
                res_type_capacity -= torch.eye(res_type_capacity.size()[1]).cuda()[torch.tensor(self.id2type).cuda()[action_idx.view(-1)]]

                assert torch.sum((money >= 0).float()) == batch_size
                assert torch.sum((res_action_capacity >= 0).float()) == len(res_action_capacity.view(-1))
                assert torch.sum((res_type_capacity >= 0).float()) == len(res_type_capacity.view(-1))
            action_list_by_category.append(utils.remove_token(torch.cat(action_list_category, 1).cpu().numpy().tolist(), self.end_idx))
            action_prob_by_category.append(torch.cat(action_prob_category, 1))
        
        action_list = torch.cat(action_list, 1).cpu().numpy().tolist()
        action_list = utils.remove_token(action_list, self.end_idx)
        
        return action_list, action_list_by_category, action_prob_by_category

    def forward(self, data, gate=True):
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        batch_size = len(data)
        h = []
        money_all = []
        x_s_all = []
        side_all = []

        for db in data:
            side, x_s, money_s, perf_s, score, x_t, x_o, x_s_history, score_s_history = db
            money_all.append(money_s[0])
            x_s_all.append(x_s)
            side_all.append(side[0])

            # represent self
            if self.shared_attention_weight:
                hs = self.low_att(self.get_embedding(x_s))
            else:
                hs = self.low_att(self.get_embedding(x_s), 's')

            # represent allies (include self)
            ht = []
            money_t = []
            for xti, moneyi, perfi in x_t:
                if self.shared_attention_weight:
                    hti = self.low_att(self.get_embedding(xti))
                else:
                    hti = self.low_att(self.get_embedding(xti), 't')
                ht.append(hti.unsqueeze(0))
                money_t.append(moneyi[0])

            ht = torch.cat(ht, 0)
            if self.shared_attention_weight:
                ht = self.high_att(ht)
            else:
                ht = self.high_att(ht, 't')

            # represent enemies
            ho = []
            for xoi, moneyi, perfi in x_o:
                if self.shared_attention_weight:
                    hoi = self.low_att(self.get_embedding(xoi))
                else:
                    hoi = self.low_att(self.get_embedding(xoi), 'o')
                #hoi = torch.cat([hoi, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
                ho.append(hoi.unsqueeze(0))

            ho = torch.cat(ho, 0)
            if self.shared_attention_weight:
                ho = self.high_att(ho)
            else:
                ho = self.high_att(ho, 'o')

            # money representation
            # teammate money only
            h_money = self.encode_money(torch.tensor(money_t).cuda())

            # incorporate round history information into self representation
            if self.hist_encoding is not None:
                h_s_history = []
                for x_s_h in x_s_history:
                    if self.shared_attention_weight:
                        X = self.low_att(self.get_embedding(x_s_h))
                        if self.different_attention_weight:
                            X = self.low_att(self.get_embedding(x_s_h), 'h')
                    else:
                        X = self.low_att(self.get_embedding(x_s_h), 's')
                    h_s_history.append(X.unsqueeze(0))
                h_s_history = torch.cat(h_s_history, 0)
            if self.hist_encoding is None:
                h_s_history = None
            else:
                hist_model = self.hist_encoding.split('.')
                if hist_model[-1] == 'time':
                    power_idx = torch.arange(h_s_history.size()[0]).float().cuda()
                    power_idx = torch.flip(power_idx, [-1])
                    w_time = self.time_decaying ** power_idx
                    w_time = w_time / torch.sum(w_time)
                    h_s_history = w_time.unsqueeze(1) * h_s_history
                if hist_model[0] == 'avg':
                    h_s_history = torch.mean(h_s_history, 0)
                elif hist_model[0] == 'score_weighted':
                    w_score = torch.tensor(score_s_history).cuda().view(-1, 1) + EPSILON
                    norm_w_score = w_score / torch.sum(w_score)
                    h_s_history = torch.sum(norm_w_score * h_s_history, 0)
                else:
                    raise NotImplementedError("Specified way to process self history is not defined.")

            # concat representations
            if self.hist_encoding is not None:
                hb = torch.cat([h_s_history, hs, ht, ho, h_money], -1)
            else:
                hb = torch.cat([hs, ht, ho, h_money], -1)

            # incorporate team information
            hb = torch.cat([hb, self.side_embedding[side[0]]], -1)

            h.append(hb.unsqueeze(0))

        h = torch.cat(h, 0)
        assert h.size()[0] == batch_size

        # seperate binary classifier
        bi_prob = []
        no_output_bi = []
        if self.lstm_mode == 'triple':
            for i in range(self.output_categories):
                h_bi = self.BiClassif(h.detach(), i)
                bi_prob.append(h_bi.unsqueeze(1))
                if gate:
                    no_output_bi.append((h_bi[:, 0] > h_bi[:, 1]).unsqueeze(1))
                else:
                    no_output_bi.append((h_bi[:, 0] != h_bi[:, 0]).unsqueeze(1))            
        elif self.lstm_mode == 'single':
            h_bi = self.BiClassif(h.detach())
            bi_prob.append(h_bi.unsqueeze(1))
            if gate:
                no_output_bi.append((h_bi[:, 0] > h_bi[:, 1]).unsqueeze(1))
            else:
                no_output_bi.append((h_bi[:, 0] != h_bi[:, 0]).unsqueeze(1))
        bi_prob = torch.cat(bi_prob, 1)
        no_output_bi = torch.cat(no_output_bi, 1)
        
        # resource left
        money = torch.tensor(money_all).cuda() * self.money_scaling
        res_action_capacity = []
        res_type_capacity = []
        for x_s in x_s_all:
            res_action_capacity.append(self.get_residual_capacity(x_s, self.action_capacity))
            res_type_capacity.append(self.get_residual_capacity(self.id2type[x_s], self.type_capacity))
        res_action_capacity = torch.tensor(res_action_capacity).cuda()
        res_type_capacity = torch.tensor(res_type_capacity).cuda()

        # initialize lstm
        init_action = [self.start_idx] * batch_size
        init_embedding = self.get_embedding(init_action).unsqueeze(1)
        # transform representation to initialize (h, c)
        init_h = self.HLN(h).view(self.history_num_layers, batch_size, self.history_dim)
        init_c = self.CLN(h).view(self.history_num_layers, batch_size, self.history_dim)

        action_list, action_list_by_category, action_prob_by_category = self.generate(init_embedding, init_h, init_c, no_output_bi, money.clone(), side_all, res_action_capacity.clone(), res_type_capacity.clone(), is_greedy=False)
        
        greedy_list, greedy_list_by_category, _ = self.generate(init_embedding, init_h, init_c, no_output_bi, money.clone(), side_all, res_action_capacity.clone(), res_type_capacity.clone(), is_greedy=True)
        
        return action_list, greedy_list, action_prob_by_category, no_output_bi.cpu().numpy().tolist(), bi_prob, action_list_by_category, greedy_list_by_category

    def initialize_lstm(self, init_embedding, init_state, category_id=None):
        if category_id is None:
            self.history = [self.rnn(init_embedding, init_state)[1]]
        elif category_id == 0:
            self.history = [self.rnn1(init_embedding, init_state)[1]]
        elif category_id == 1:
            self.history = [self.rnn2(init_embedding, init_state)[1]]
        elif category_id == 2:
            self.history = [self.rnn3(init_embedding, init_state)[1]]        
        else:
            raise NotImplementedError("Category ID exceeds number of output categories.")

    def update_lstm(self, action, category_id=None, offset=None):

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        embedding = self.get_embedding(action).view(-1, 1, self.embedding_dim)
        if offset is not None:
            offset_path_history(self.history, offset.view(-1))
        torch.backends.cudnn.enabled = False
        if category_id is None:
            self.history.append(self.rnn(embedding, self.history[-1])[1])
        elif category_id == 0:
            self.history.append(self.rnn1(embedding, self.history[-1])[1])
        elif category_id == 1:
            self.history.append(self.rnn2(embedding, self.history[-1])[1])
        elif category_id == 2:
            self.history.append(self.rnn3(embedding, self.history[-1])[1])
        else:
            raise NotImplementedError("Category ID exceeds number of output categories.")

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    # This method is not used.
    def predict(self, data, gate=True):
        '''
        x: input
        '''
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        batch_size = len(data)
        h = []
        money_all = []
        x_s_all = []
        side_all = []

        for db in data:
            side, x_s, money_s, perf_s, score, x_t, x_o = db
            money_all.append(money_s[0])
            x_s_all.append(x_s)
            side_all.append(side[0])
            # represent allies
            ht = []
            for xti, moneyi, perfi in x_t:
                if self.shared_attention_weight:
                    hti = self.low_att(self.get_embedding(xti))
                else:
                    hti = self.low_att(self.get_embedding(xti), 't')
                hti = torch.cat([hti, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
                ht.append(hti.unsqueeze(0))
            ht = torch.cat(ht, 0)
            if self.shared_attention_weight:
                ht = self.high_att(ht)
            else:
                ht = self.high_att(ht, 't')

            # represent enemies
            ho = []
            for xoi, moneyi, perfi in x_o:
                if self.shared_attention_weight:
                    hoi = self.low_att(self.get_embedding(xoi))
                else:
                    hoi = self.low_att(self.get_embedding(xoi), 'o')
                hoi = torch.cat([hoi, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
                ho.append(hoi.unsqueeze(0))
            ho = torch.cat(ho, 0)
            if self.shared_attention_weight:
                ho = self.high_att(ho)
            else:
                ho = self.high_att(ho, 'o')

            # represent self
            if self.shared_attention_weight:
                hs = self.low_att(self.get_embedding(x_s))
            else:
                hs = self.low_att(self.get_embedding(x_s), 's')
            hs = torch.cat([hs, torch.tensor(money_s).cuda(), torch.tensor(perf_s).cuda()], -1)


            # concat representations
            hb = torch.cat([hs, ht, ho], -1)

            # incorporate team information
            hb = torch.cat([hb, self.side_embedding[side[0]], torch.tensor(score).cuda()], -1)

            h.append(hb.unsqueeze(0))

        h = torch.cat(h, 0)
        assert h.size()[0] == batch_size

        # seperate binary classifier
        bi_prob = []
        no_output_bi = []
        for i in range(self.output_categories):
            h_bi = self.BiClassif(h.detach(), i)
            bi_prob.append(h_bi.unsqueeze(1))
            no_output_bi.append((h_bi[:, 0] > h_bi[:, 1]).unsqueeze(1))
        bi_prob = torch.cat(bi_prob, 1)
        no_output_bi = torch.cat(no_output_bi, 1)

        # return values - predictions and probabilities
        action_list = None

        # resource left
        money = torch.tensor(money_all).cuda() * self.money_scaling
        res_action_capacity = []
        res_type_capacity = []
        for x_s in x_s_all:
            res_action_capacity.append(self.get_residual_capacity(x_s, self.action_capacity))
            res_type_capacity.append(self.get_residual_capacity(self.id2type[x_s], self.type_capacity))
        res_action_capacity = torch.tensor(res_action_capacity).cuda()
        res_type_capacity = torch.tensor(res_type_capacity).cuda()


        # transform representation to initialize (h, c)
        init_h = self.HLN(h).view(self.history_num_layers, batch_size, self.history_dim)
        init_c = self.CLN(h).view(self.history_num_layers, batch_size, self.history_dim)
        side_all = torch.tensor(side_all).cuda()

        # beam search
        log_action_prob = torch.zeros(1).cuda()
        k = 1
        for j in range(self.output_categories):

            # initialize LSTM
            init_action = [self.start_idx] * batch_size * k
            init_embedding = self.get_embedding(init_action).unsqueeze(1)
            init_h_tile = utils.tile_along_beam(init_h, k, 1)
            init_c_tile = utils.tile_along_beam(init_c, k, 1)
            self.initialize_lstm(init_embedding, (init_h_tile, init_c_tile), j)
            # TODO: is_end tile: (batch_size,) - > (batch_size * last_k)
            is_end = utils.tile_along_beam(no_output_bi[:, j].float(), k, -1)


            for i in range(self.max_output_num):
                if torch.sum(is_end) == len(is_end):
                    for _ in range(self.max_output_num - i):
                        if action_list is not None:
                            action_list = torch.cat([action_list, torch.ones(batch_size * k, 1).long().cuda() * self.end_idx], dim=1)
                        else:
                            action_list = torch.ones(batch_size * k, 1).long().cuda() * self.end_idx
                    break

                H = self.history[-1][0][-1, :, :]
                action_dist = self.classif_LN(H, j)

                side_all = utils.tile_along_beam(side_all.view(batch_size, -1)[:, 0], k, 0)
                action_mask = self.money_mask(money.unsqueeze(1)) * self.side_mask[side_all]
                #action_mask *= self.get_capacity_mask(res_action_capacity, self.mute_action_mask)
                action_mask *= (res_action_capacity > 0).float()
                action_mask *= self.get_capacity_mask(res_type_capacity, self.mute_type_mask)
                action_mask = action_mask[:, self.category_action_offset[j]]
                action_dist = action_dist * action_mask

                mask_all_tensor = torch.zeros(action_dist.size()[1], dtype = torch.float).cuda()
                mask_all_tensor[-1] = 1.0

                action_dist = (1 - is_end.unsqueeze(1)) * action_dist + is_end.unsqueeze(1) * mask_all_tensor

                log_action_dist = log_action_prob.view(-1, 1) + torch.log(action_dist)
                assert log_action_dist.size()[1] == self.output_dims[j]
                log_action_dist = log_action_dist.view(batch_size, -1)

                last_k = k
                k = min(self.beam_size, log_action_dist.size()[1])
                k = 1
                log_action_prob, action_ind = torch.topk(log_action_dist, k)
                action_offset = (action_ind / self.output_dims[j] + torch.arange(batch_size).unsqueeze(1).cuda() * last_k).view(-1)

                output_idx = action_ind % self.output_dims[j]
                action_idx = torch.tensor(self.category_action_offset[j]).cuda()[output_idx].view(-1)

                is_end = (action_idx == self.end_idx).view(-1).float()

                # TODO: update action_list, hidden_state
                if action_list is not None:
                    action_list = torch.cat([action_list[action_offset], action_idx.view(-1, 1)], dim=1)
                else:
                    action_list = action_idx.view(-1, 1)

                self.update_lstm(action_idx, j, offset=action_offset)

                money = money[action_offset] - self.prices[action_idx]
                res_action_capacity = res_action_capacity[action_offset]
                res_action_capacity -= torch.eye(res_action_capacity.size()[1]).cuda()[action_idx]
                res_type_capacity = res_type_capacity[action_offset]
                res_type_capacity -= torch.eye(res_type_capacity.size()[1]).cuda()[torch.tensor(self.id2type).cuda()[action_idx]]

                '''assert torch.sum((money >= 0).float()) == batch_size * k
                assert torch.sum((res_action_capacity >= 0).float()) == len(res_action_capacity.view(-1))
                assert torch.sum((res_type_capacity >= 0).float()) == len(res_type_capacity.view(-1))'''


            # TODO: outer greedy, inner beam search
            '''money = money.view(batch_size, -1)[:, 0]
            res_action_capacity = res_action_capacity.view(batch_size, -1, res_action_capacity.size()[1])[:, 0, :]
            res_type_capacity = res_type_capacity.view(batch_size, -1, res_type_capacity.size()[1])[:, 0, :]'''

        #action_list = torch.cat(action_list, 1).cpu().numpy().tolist()
        assert len(action_list) == batch_size * k
        action_list = action_list.view(batch_size, k, -1)[:, 0].cpu().numpy().tolist()
        action_list = utils.remove_token(action_list, self.end_idx)
        action_log_prob = log_action_prob.view(batch_size, k)[:, -1].cpu().detach().numpy().tolist()

        return action_list, action_log_prob, no_output_bi.cpu().numpy().tolist()

    def define_modules(self):
        # terrorist
        if self.shared_attention_weight:
            self.att_LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1 = nn.Linear(self.ff_dim, 1)
            #self.att_LN2 = nn.Linear(self.embedding_dim + self.resource_dim, self.ff_dim)
            self.att_LN2 = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v2 = nn. Linear(self.ff_dim, 1)
            if self.different_attention_weight:
                self.att_LN1_h = nn.Linear(self.embedding_dim, self.ff_dim)
                self.v1_h = nn.Linear(self.ff_dim, 1)
        else:
            self.att_LN1_s = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_s = nn.Linear(self.ff_dim, 1)
            self.att_LN1_t = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_t = nn.Linear(self.ff_dim, 1)
            self.att_LN1_o = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_o = nn.Linear(self.ff_dim, 1)

            self.att_LN2_t = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v2_t = nn. Linear(self.ff_dim, 1)
            self.att_LN2_o = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v2_o = nn. Linear(self.ff_dim, 1)

        self.HLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)
        self.CLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)

        self.LNDropout = nn.Dropout(p=self.ff_dropout_rate)

        assert self.output_categories == 3

        self.money_LN1 = nn.Linear(self.money_dim, self.ff_dim)
        self.money_LN2 = nn.Linear(self.ff_dim, self.money_dim)
        
        if self.lstm_mode == 'triple':
            self.BClassif1_1=nn.Linear(self.input_dim, self.ff_dim)
            self.BClassif1_2=nn.Linear(self.ff_dim, 2 * self.ff_dim)
            self.BClassif1_3=nn.Linear(2 * self.ff_dim, self.ff_dim)
            self.BClassif1_4=nn.Linear(self.ff_dim, 2)

            self.BClassif2_1=nn.Linear(self.input_dim, self.ff_dim)
            self.BClassif2_2=nn.Linear(self.ff_dim, 2 * self.ff_dim)
            self.BClassif2_3=nn.Linear(2 * self.ff_dim, self.ff_dim)
            self.BClassif2_4=nn.Linear(self.ff_dim, 2)

            self.BClassif3_1=nn.Linear(self.input_dim, self.ff_dim)
            self.BClassif3_2=nn.Linear(self.ff_dim, 2 * self.ff_dim)
            self.BClassif3_3=nn.Linear(2 * self.ff_dim, self.ff_dim)
            self.BClassif3_4=nn.Linear(self.ff_dim, 2)
        
            self.rnn1 = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.history_dim,
                                num_layers = self.history_num_layers,
                                batch_first = True)

            self.rnn2 = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.history_dim,
                                num_layers = self.history_num_layers,
                                batch_first = True)

            self.rnn3 = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.history_dim,
                                num_layers = self.history_num_layers,
                                batch_first = True)

            self.LN1_1 = nn.Linear(self.history_dim, self.ff_dim)
            self.LN1_2 = nn.Linear(self.ff_dim, self.output_dim1)

            self.LN2_1 = nn.Linear(self.history_dim, self.ff_dim)
            self.LN2_2 = nn.Linear(self.ff_dim, self.output_dim2)

            self.LN3_1 = nn.Linear(self.history_dim, self.ff_dim)
            self.LN3_2 = nn.Linear(self.ff_dim, self.output_dim3)        
        elif self.lstm_mode == 'single':
            self.BClassif1=nn.Linear(self.input_dim, self.ff_dim)
            self.BClassif2=nn.Linear(self.ff_dim, 2 * self.ff_dim)
            self.BClassif3=nn.Linear(2 * self.ff_dim, self.ff_dim)
            self.BClassif4=nn.Linear(self.ff_dim, 2)

        
            self.rnn = nn.LSTM(input_size = self.embedding_dim,
                                hidden_size = self.history_dim,
                                num_layers = self.history_num_layers,
                                batch_first = True)
            self.LN1 = nn.Linear(self.history_dim, self.ff_dim)
            self.LN2 = nn.Linear(self.ff_dim, self.output_dim)

        if torch.cuda.is_available():
            if self.shared_attention_weight:
                self.att_LN1 = self.att_LN1.cuda()
                self.v1 = self.v1.cuda()
                self.att_LN2 = self.att_LN2.cuda()
                self.v2 = self.v2.cuda()
                if self.different_attention_weight:
                    self.att_LN1_h = self.att_LN1_h.cuda()
                    self.v1_h = self.v1_h.cuda()
            else:
                self.att_LN1_s = self.att_LN1_s.cuda()
                self.v1_s = self.v1_s.cuda()
                self.att_LN1_t = self.att_LN1_t.cuda()
                self.v1_t = self.v1_t.cuda()
                self.att_LN1_o = self.att_LN1_o.cuda()
                self.v1_o = self.v1_o.cuda()
                self.att_LN2_t = self.att_LN2_t.cuda()
                self.v2_t = self.v2_t.cuda()
                self.att_LN2_o = self.att_LN2_o.cuda()
                self.v2_o = self.v2_o.cuda()

            self.HLN = self.HLN.cuda()
            self.CLN = self.CLN.cuda()
            self.LNDropout = self.LNDropout.cuda()
            
            if self.lstm_mode == 'triple':
                self.BClassif1_1=self.BClassif1_1.cuda()
                self.BClassif1_2=self.BClassif1_2.cuda()
                self.BClassif1_3=self.BClassif1_3.cuda()
                self.BClassif1_4=self.BClassif1_4.cuda()

                self.BClassif2_1=self.BClassif2_1.cuda()
                self.BClassif2_2=self.BClassif2_2.cuda()
                self.BClassif2_3=self.BClassif2_3.cuda()
                self.BClassif2_4=self.BClassif2_4.cuda()

                self.BClassif3_1=self.BClassif3_1.cuda()
                self.BClassif3_2=self.BClassif3_2.cuda()
                self.BClassif3_3=self.BClassif3_3.cuda()
                self.BClassif3_4=self.BClassif3_4.cuda()
            
                self.rnn1 = self.rnn1.cuda()
                self.rnn2 = self.rnn2.cuda()
                self.rnn3 = self.rnn3.cuda()

                self.LN1_1 = self.LN1_1.cuda()
                self.LN1_2 = self.LN1_2.cuda()

                self.LN2_1 = self.LN2_1.cuda()
                self.LN2_2 = self.LN2_2.cuda()

                self.LN3_1 = self.LN3_1.cuda()
                self.LN3_2 = self.LN3_2.cuda()
            elif self.lstm_mode == 'single':
                self.BClassif1=self.BClassif1.cuda()
                self.BClassif2=self.BClassif2.cuda()
                self.BClassif3=self.BClassif3.cuda()
                self.BClassif4=self.BClassif4.cuda()

                self.rnn = self.rnn.cuda()
                self.LN1 = self.LN1.cuda()
                self.LN2 = self.LN2.cuda()

    def initialize_modules(self):
        # xavier initialization
        if self.lstm_mode == 'triple':
            for name, param in self.rnn1.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

            for name, param in self.rnn2.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

            for name, param in self.rnn3.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        elif self.lstm_mode == 'single':
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

        if self.shared_attention_weight:
            nn.init.xavier_uniform_(self.att_LN1.weight)
            nn.init.xavier_uniform_(self.att_LN2.weight)
            nn.init.xavier_uniform_(self.v1.weight)
            nn.init.xavier_uniform_(self.v2.weight)
            if self.different_attention_weight:
                nn.init.xavier_uniform_(self.att_LN1_h.weight)
                nn.init.xavier_uniform_(self.v1_h.weight)
        else:
            nn.init.xavier_uniform_(self.att_LN1_s.weight)
            nn.init.xavier_uniform_(self.v1_s.weight)
            nn.init.xavier_uniform_(self.att_LN1_t.weight)
            nn.init.xavier_uniform_(self.v1_t.weight)
            nn.init.xavier_uniform_(self.att_LN1_o.weight)
            nn.init.xavier_uniform_(self.v1_o.weight)
            nn.init.xavier_uniform_(self.att_LN2_t.weight)
            nn.init.xavier_uniform_(self.v2_t.weight)
            nn.init.xavier_uniform_(self.att_LN2_o.weight)
            nn.init.xavier_uniform_(self.v2_o.weight)

        nn.init.xavier_uniform_(self.HLN.weight)
        nn.init.xavier_uniform_(self.CLN.weight)

        if self.lstm_mode == 'triple':
            nn.init.xavier_uniform_(self.BClassif1_1.weight)
            nn.init.xavier_uniform_(self.BClassif1_2.weight)
            nn.init.xavier_uniform_(self.BClassif1_3.weight)
            nn.init.xavier_uniform_(self.BClassif1_4.weight)

            nn.init.xavier_uniform_(self.BClassif2_1.weight)
            nn.init.xavier_uniform_(self.BClassif2_2.weight)
            nn.init.xavier_uniform_(self.BClassif2_3.weight)
            nn.init.xavier_uniform_(self.BClassif2_4.weight)

            nn.init.xavier_uniform_(self.BClassif3_1.weight)
            nn.init.xavier_uniform_(self.BClassif3_2.weight)
            nn.init.xavier_uniform_(self.BClassif3_3.weight)
            nn.init.xavier_uniform_(self.BClassif3_4.weight)

            nn.init.xavier_uniform_(self.LN1_1.weight)
            nn.init.xavier_uniform_(self.LN1_2.weight)

            nn.init.xavier_uniform_(self.LN2_1.weight)
            nn.init.xavier_uniform_(self.LN2_2.weight)

            nn.init.xavier_uniform_(self.LN3_1.weight)
            nn.init.xavier_uniform_(self.LN3_2.weight)
        elif self.lstm_mode == 'single':
            nn.init.xavier_uniform_(self.BClassif1.weight)
            nn.init.xavier_uniform_(self.BClassif2.weight)
            nn.init.xavier_uniform_(self.BClassif3.weight)
            nn.init.xavier_uniform_(self.BClassif4.weight)

            nn.init.xavier_uniform_(self.LN1.weight)
            nn.init.xavier_uniform_(self.LN2.weight)

    def clone(self, npy_dict):
        clone = CsgoModel(self.args, npy_dict)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone
