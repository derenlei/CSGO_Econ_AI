import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import Counter
import src.utils as utils
import time

import argparse


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
        # ReptileModel.__init__(self)
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
        
        
        # all actions in all lstm output list
        '''self.category_mask = torch.zeros(self.output_categories, len(self.id2type)).float().cuda()
        for i, typeid in enumerate(self.id2type):
            if typeid == 0:
                self.category_mask[0][i] = 1.0
            elif 1 <= typeid <= 5:
                self.category_mask[0][i] = 1.0
            elif typeid == 6:
                self.category_mask[1][i] = 1.0
            else:
                self.category_mask[2][i] = 1.0
        # do not mask end token for every category
        self.category_mask[0][self.end_idx] = 1.0
        self.category_mask[1][self.end_idx] = 1.0
        self.category_mask[2][self.end_idx] = 1.0
        #self.category_mask[3][self.end_idx] = 1.0'''
        
        self.money_scaling = args.money_scaling
        
        # self.action_mask = torch.ones(self.output_dim).float().cuda()
        # self.action_mask[self.start_idx] = 0.0
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
        self.team_dim = 4
#         self.team_dim = 2  # no round score info
        self.input_dim = (self.embedding_dim + self.resource_dim) * 3 + self.team_dim
#         self.input_dim = (self.embedding_dim + self.resource_dim) * 2 + self.team_dim  # no teammate
        
        self.ff_dropout_rate = args.ff_dropout_rate
        self.max_output_num = args.max_output_num
        self.beam_size = args.beam_size
        self.shared_attention_weight = args.shared_attention_weight

        self.define_modules()
        #xavier_initialization
        self.initialize_modules()
        
        
    def reward_fun(self, a, a_r):
        # F1 score
        # remove end token
        '''if a[-1] == self.end_idx:
            a_new = a[: -1]
        else:
            a_new = a'''
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
        #global debug_loss, last_log_prob
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
        
#         loss_start_time = time.time()
        
        batch_size = len(action_list)
        
        no_output_bi = torch.tensor(no_output_bi).cuda()

        '''log_action_probs = []
        for ap in action_prob:
            log_action_probs.append(torch.log(ap).unsqueeze(0))
        # all the gates are closed
        if len(log_action_probs) == 0:
            log_action_probs = torch.zeros(1).double().cuda()
        else:
            log_action_probs = torch.cat(log_action_probs, 0)'''
        
        '''log_bi_probs = []
        for bp in bi_prob:
            log_bi_probs.append(torch.log(bp).unsqueeze(0))
        log_bi_probs = torch.cat(log_bi_probs, 0)'''
        log_bi_probs = torch.log(bi_prob)
        
        #gs_actions = labels
        #bi_labels = torch.tensor(utils.get_category_label(labels, self.id2type)).cuda().float()
        bi_labels = torch.tensor(utils.get_batched_category_label(labels, self.id2type)).cuda().float()
        
#         cat_time = time.time()
#         print(cat_time - loss_start_time)

        # Compute policy gradient loss
        # Compute discounted reward
        #final_reward = self.reward_fun(action_list, gs_actions) - self.reward_fun(greedy_list, gs_actions)
        '''if self.baseline != 'n/a':
            #print('stablized reward')
            final_reward = stablize_reward(final_reward)'''
        
        '''cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)'''
        #pg_loss = -final_reward * torch.sum(log_action_probs)
        # LSTM loss
        seq_loss_print = []
        seq_loss = torch.tensor(0.0).cuda()
        # label_by_category = utils.filter_category_actions(labels, self.id2type, self.output_categories)
        label_by_batch = utils.filter_batched_category_actions(labels, self.id2type, self.output_categories)
        
        label_by_category = utils.reshape_batched_category_actions(label_by_batch)
        
#         filter_time = time.time()
#         print("filter time:", filter_time - cat_time)
        
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
        seq_loss /= batch_size
        
#         seq_time = time.time()
#         print("seq time:", seq_time - filter_time)
        '''# debug
        debug_loss = []
        last_log_prob = []'''
        
        '''for i in range(self.output_categories):
            if no_output_bi[i]:
                seq_loss_print.append(np.nan)
            else:
                reward_sample = self.reward_fun(action_list_by_category[i], label_by_category[i])
                reward_greedy = self.reward_fun(greedy_list_by_category[i], label_by_category[i])
                reward = reward_sample - reward_greedy
                log_prob = torch.sum(torch.log(action_prob_by_category[i]))
                seq_loss_category = -reward * log_prob
                
#                 print('reward_sample', reward_sample)
#                 print('reward_greedy', reward_greedy)
#                 print("reward:", reward)
#                 print("log_prob:", -log_prob)
                
                seq_loss += seq_loss_category
                seq_loss_print.append(seq_loss_category.detach().item())
                #debug_loss.append(seq_loss_category.detach())
                #last_log_prob.append(log_prob.data.cpu())'''
        
        # binary classifier loss
        #bi_loss = -torch.sum(log_bi_probs * bi_labels)
        #bi_loss_print = list(-torch.sum(log_bi_probs * bi_labels, 1).detach().cpu().numpy())
        bi_loss = -torch.sum(log_bi_probs * bi_labels) / batch_size
        bi_loss_print = list((torch.sum((log_bi_probs * bi_labels).view(self.output_categories, -1), 1) / batch_size).cpu().detach().numpy())
        
#         bi_time = time.time()
#         print("bi time:", bi_time - seq_time)
        
        #debug_loss.append(bi_loss.detach())
        '''print(-torch.sum(log_bi_probs * bi_labels, 1))
        print(-torch.sum(log_bi_probs * bi_labels, 1).detach())
        print(list(-torch.sum(log_bi_probs * bi_labels, 1).detach().cpu().numpy()))
        print(type(np.array([1,0,1])))
        print(type(bi_loss_print))
        print(bi_loss_print.shape)
        assert 1==0'''
        #bi_loss = - torch.sum(log_bi_probs[0] * bi_labels[0])

        #########################
        # SWITCH pytorch BCELoss#
        #########################
        '''
        criteria = nn.BCELoss()
        bi_loss = criteria(bi_prob[0],bi_labels[0])'''
        
        #pt_loss = -final_reward * torch.sum(torch.exp(log_action_probs))
        
        loss_dict = {}
        loss_dict['model_loss'] = seq_loss.double() + bi_loss.double()
        '''if torch.isnan(loss_dict['model_loss']):
            print(debug_loss)
            assert 1==0'''
        loss_dict['bi_loss'] = bi_loss_print
        loss_dict['seq_loss'] = seq_loss_print
        #loss_dict['print_loss'] = float(pt_loss)
        
        #loss_dict['reward'] = final_reward
        # loss_dict['entropy'] = float(entropy.mean())
        
        return loss_dict
    
    def get_embedding(self, idx):
        '''if not isinstance(idx, list):
            return self.embedding[idx]
        else:
            ret = []
            for i in idx:
                ret.append(self.embedding[i])
            return ret'''
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
#         print("high att:", att.detach().cpu().numpy().tolist())
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
        h = torch.tanh(h)
        if side is None:
            h = self.v1(h)
        elif side == 's':
            h = self.v1_s(h)
        elif side == 't':
            h = self.v1_t(h)
        elif side == 'o':
            h = self.v1_o(h)
        att = F.softmax(h, 0)
        ret = torch.sum(att * x, 0)
        return ret
    
    def BiClassif(self, x, category_id):
        if category_id == 0:
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
    
    def classif_LN(self, x, category_id):
        if category_id == 0:
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
        #action_dist = F.softmax(
        #    torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
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
        # is_mute = (torch.tensor(res_action_capacity) == 0).float().cuda()
#         ret = []
#         for j in range(len(res_action_capacity)):
#             mask = torch.ones(mute_mask.size()[1]).cuda()
#             for i, res_cap in enumerate(res_action_capacity[j]):
#                 if res_cap == 0:
#                     mask *= mute_mask[i]
#             ret.append(mask.unsqueeze(0))
#         return torch.cat(ret, 0)
        to_multiply_mask = 1 - (res_capacity > 0).float()
        ret = (torch.mm(to_multiply_mask, mute_mask) == torch.sum(to_multiply_mask, 1).unsqueeze(1)).float()
        return ret
        
    
    def forward(self, data, gate=True):
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
#         start_time = time.time()
        
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
#             hb = torch.cat([hs, ho], -1)  # no teammate

            # incorporate team information
            hb = torch.cat([hb, self.side_embedding[side[0]], torch.tensor(score).cuda()], -1)
#             hb = torch.cat([hb, self.side_embedding[side[0]]], -1) # no round score info
            
            h.append(hb.unsqueeze(0))
        
        h = torch.cat(h, 0)
        assert h.size()[0] == batch_size
        
#         encoding_time = time.time()
#         print("encoding time:", encoding_time - start_time)
        
        '''# return binary classifier values
        bi_prob = []
        no_output_bi = []
        # binary classifier for each category indicating whether to generate actions in this category
        for i in range(self.output_categories):
            break
            # do not backward binary classification loss
            #h_bi = self.BiClassif1[i](h.detach())
            h_bi = self.BiClassif1[i](h)
            h_bi = F.relu(h_bi)
            # h_bi = F.relu(h_bi)
            #h_bi = self.LNDropout(h_bi)
            h_bi = self.BiClassif2[i](h_bi)
            h_bi = F.relu(h_bi)
            #h_bi = self.LNDropout(h_bi)
            h_bi = self.BiClassif3[i](h_bi)
            h_bi = F.relu(h_bi)
            #h_bi = self.LNDropout(h_bi)
            h_bi = self.BiClassif4[i](h_bi)
            h_bi = F.softmax(h_bi, dim = -1).view(-1)
            bi_prob.append(h_bi)
            no_output_bi.append(h_bi[0] > h_bi[1])'''
        
        # seperate binary classifier
        bi_prob = []
        no_output_bi = []
        for i in range(self.output_categories):
            h_bi = self.BiClassif(h.detach(), i)
            bi_prob.append(h_bi.unsqueeze(1))
            no_output_bi.append((h_bi[:, 0] > h_bi[:, 1]).unsqueeze(1))
        bi_prob = torch.cat(bi_prob, 1)
        no_output_bi = torch.cat(no_output_bi, 1)
        
#         bi_time = time.time()
#         print("bi classifier:", bi_time - encoding_time)
        
        # return values - predictions and probabilities
        action_list = []
        greedy_list = []
        action_prob_by_category = []
        action_list_by_category = []
        greedy_list_by_category = []
        
        # resource left
        money = torch.tensor(money_all).cuda() * self.money_scaling
        res_action_capacity = []
        res_type_capacity = []
        for x_s in x_s_all:
            res_action_capacity.append(self.get_residual_capacity(x_s, self.action_capacity))
            res_type_capacity.append(self.get_residual_capacity(self.id2type[x_s], self.type_capacity))
        #res_action_capacity = self.get_residual_capacity(x_s_all, self.action_capacity)
        #res_type_capacity = self.get_residual_capacity(self.id2type[x_s], self.type_capacity)
        res_action_capacity = torch.tensor(res_action_capacity).cuda()
        res_type_capacity = torch.tensor(res_type_capacity).cuda()
        
        # initialize lstm
        init_action = [self.start_idx] * batch_size
        init_embedding = self.get_embedding(init_action).unsqueeze(1)
        # transform representation to initialize (h, c)
        init_h = self.HLN(h).view(self.history_num_layers, batch_size, self.history_dim)
        init_c = self.CLN(h).view(self.history_num_layers, batch_size, self.history_dim)
        '''init_h = self.HLN(h.detach()).view(self.history_num_layers, 1, self.history_dim)
        init_c = self.CLN(h.detach()).view(self.history_num_layers, 1, self.history_dim)'''
        
        # debug
        #print(self.att_LN1.weight.grad)
        '''for name, param in self.rnn1.named_parameters():
            print(name)
            print(param)
        assert 1==0
        global last_weight, last_weight_grad, action_dist_debug1, action_dist_debug2, action_dist_debug3
        if torch.isnan(torch.sum(self.att_LN1.weight)):
            print(debug_loss)
            print(last_weight)
            print(last_weight_grad)
            print(self.att_LN1.weight)
            print(self.att_LN1.weight.grad)
            print(self.rnn1.weight_ih_l0)
            print(self.rnn2.weight_ih_l0)
            print(self.rnn3.weight_ih_l0)
            print(last_log_prob)
            print(last_log_prob[0].grad)
            print(action_dist_debug1)
            print(action_dist_debug2)
            print(action_dist_debug3)
            assert 1==0
        
        last_weight = self.att_LN1.weight.clone()
        last_weight_grad = self.att_LN1.weight.grad
        action_dist_debug1, action_dist_debug2, action_dist_debug3 = [], [], []'''
#         ls_time = time.time()
#         print("lstm preprocess time:", ls_time - bi_time)
        
        # generate predictions
        for j in range(self.output_categories):
            '''if gate and no_output_bi[j]:
                action_prob_by_category.append([])
                action_list_by_category.append([])
                #print('no output')
                continue
            else:'''
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

                #history_temp = self.history
                #H_temp = H.cpu()
                #action_dist_unmask = action_dist.cpu()

                action_mask = self.money_mask(money.unsqueeze(1)) * self.side_mask[side_all]
                #action_mask *= self.category_mask[j]
                #action_mask *= self.get_capacity_mask(res_action_capacity, self.mute_action_mask)
                action_mask *= (res_action_capacity > 0).float()
                action_mask *= self.get_capacity_mask(res_type_capacity, self.mute_type_mask)
                action_mask = action_mask[:, self.category_action_offset[j]]
                action_dist = action_dist * action_mask # + (1 - action_mask) * EPSILON

                # nan in action_dist
                #action_dist[action_dist != action_dist] = 0.0

                is_zero = (torch.sum(action_dist, 1) == 0).float().unsqueeze(1)
                mask_all_tensor = torch.zeros(action_dist.size()[1], dtype = torch.float).cuda()
                mask_all_tensor[-1] = 1.0
                action_dist = action_dist + is_zero * mask_all_tensor
                
                if is_end is not None:
                    action_dist = (1 - is_end.unsqueeze(1)) * action_dist + is_end.unsqueeze(1) * mask_all_tensor
                
                # normalize
                action_dist = action_dist / torch.sum(action_dist, 1).unsqueeze(1)
                
                output_idx = torch.multinomial(action_dist, 1, replacement=True)
                action_idx = torch.tensor(self.category_action_offset[j]).cuda()[output_idx]
                
                #action_prob.append(action_dist[0][action_idx])
                #action_prob_category.append(action_dist[0][output_idx].unsqueeze(0))
                action_prob_category.append(torch.gather(action_dist, 1, output_idx))
                
                #if action_idx == self.end_idx:
                #    break
                # TODO: current implementation is to keep outputing end token
                is_end = (action_idx == self.end_idx).view(-1).float()
                
                #action_list.append(action_idx)
                #action_list_category.append(action_idx)
                #action_list.append(action_idx)
                action_list.append(action_idx)
                action_list_category.append(action_idx)
                
                self.update_lstm(action_idx, j)
                
                #money = money - self.prices[action_idx]
                #res_action_capacity[action_idx] -= 1
                #res_type_capacity[self.id2type[action_idx]] -= 1
                money = money - self.prices[action_idx.view(-1)]
                res_action_capacity -= torch.eye(res_action_capacity.size()[1]).cuda()[action_idx.view(-1)]
                res_type_capacity -= torch.eye(res_type_capacity.size()[1]).cuda()[torch.tensor(self.id2type).cuda()[action_idx.view(-1)]]
                
                #assert money >= 0
                #assert res_action_capacity[action_idx] >= 0
                #assert res_type_capacity[self.id2type[action_idx]] >= 0
                assert torch.sum((money >= 0).float()) == batch_size
                assert torch.sum((res_action_capacity >= 0).float()) == len(res_action_capacity.view(-1))
                assert torch.sum((res_type_capacity >= 0).float()) == len(res_type_capacity.view(-1))
                
                # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
            action_list_by_category.append(utils.remove_token(torch.cat(action_list_category, 1).cpu().numpy().tolist(), self.end_idx))
            action_prob_by_category.append(torch.cat(action_prob_category, 1))
        action_list = torch.cat(action_list, 1).cpu().numpy().tolist()
        action_list = utils.remove_token(action_list, self.end_idx)
        
        '''for j in range(self.output_categories):
            if gate and no_output_bi[j]:
                action_prob_by_category.append([])
                action_list_by_category.append([])
                #print('no output')
                continue
            else:
                # initialize LSTM
                self.initialize_lstm(init_embedding, (init_h, init_c), j)
                
                action_prob_category = []
                action_list_category = []
                for i in range(self.max_output_num):
                    H = self.history[-1][0][-1, :, :]
                    action_dist = self.classif_LN(H, j)
                    
                    #history_temp = self.history
                    #H_temp = H.cpu()
                    #action_dist_unmask = action_dist.cpu()
                    
                    action_mask = self.money_mask(money) * self.side_mask[side[0]]
                    #action_mask *= self.category_mask[j]
                    action_mask *= self.get_capacity_mask(torch.tensor(res_action_capacity).unsqueeze(0).cuda(), self.mute_action_mask).view(-1)
                    action_mask *= self.get_capacity_mask(torch.tensor(res_type_capacity).unsqueeze(0).cuda(), self.mute_type_mask).view(-1)
                    action_mask = action_mask[self.category_action_offset[j]]
                    action_dist = action_dist * action_mask # + (1 - action_mask) * EPSILON
                    
                    # nan in action_dist
                    #action_dist[action_dist != action_dist] = 0.0
                    
                    is_zero = (torch.sum(action_dist) == 0).float()
                    
                    mask_all_tensor = torch.zeros_like(action_dist, dtype = torch.float).cuda().view(-1)
                    mask_all_tensor[-1] = 1.0
                    action_dist = action_dist + is_zero * mask_all_tensor
                    # normalize
                    #action_dist = action_dist / torch.sum(action_dist)
                    output_idx = torch.multinomial(action_dist, 1, replacement=True).item()
                    action_idx = self.category_action_offset[j][output_idx]
                    
                    #action_prob.append(action_dist[0][action_idx])
                    action_prob_category.append(action_dist[0][output_idx].unsqueeze(0))
                    if action_idx == self.end_idx:
                        break
                    #action_list.append(action_idx)
                    action_list_category.append(action_idx)
                    action_list.append(action_idx)
                    self.update_lstm(action_idx, j)
                    money = money - self.prices[action_idx]
                    res_action_capacity[action_idx] -= 1
                    res_type_capacity[self.id2type[action_idx]] -= 1
                    assert money >= 0
                    assert res_action_capacity[action_idx] >= 0
                    assert res_type_capacity[self.id2type[action_idx]] >= 0
                    # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
                action_list_by_category.append(action_list_category)
                action_prob_by_category.append(torch.cat(action_prob_category, 0))'''
        
#         sample_time = time.time()
#         print("lstm sample:", sample_time - ls_time)
        
        # greedy
        # resource left
        money = torch.tensor(money_all).cuda() * self.money_scaling
        res_action_capacity = []
        res_type_capacity = []
        for x_s in x_s_all:
            res_action_capacity.append(self.get_residual_capacity(x_s, self.action_capacity))
            res_type_capacity.append(self.get_residual_capacity(self.id2type[x_s], self.type_capacity))
        #res_action_capacity = self.get_residual_capacity(x_s_all, self.action_capacity)
        #res_type_capacity = self.get_residual_capacity(self.id2type[x_s], self.type_capacity)
        res_action_capacity = torch.tensor(res_action_capacity).cuda()
        res_type_capacity = torch.tensor(res_type_capacity).cuda()
        
        for j in range(self.output_categories):
            # initialize LSTM
            self.initialize_lstm(init_embedding, (init_h, init_c), j)

            greedy_list_category = []
            is_end = no_output_bi[:, j].float()
            for i in range(self.max_output_num):
                if torch.sum(is_end) == len(is_end):
                    for _ in range(self.max_output_num - i):
                        greedy_list.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                        greedy_list_category.append(torch.ones(batch_size, 1).long().cuda() * self.end_idx)
                    break
                
                H = self.history[-1][0][-1, :, :]
                action_dist = self.classif_LN(H, j)

                action_mask = self.money_mask(money.unsqueeze(1)) * self.side_mask[side_all]
                #action_mask *= self.category_mask[j]
                #action_mask *= self.get_capacity_mask(res_action_capacity, self.mute_action_mask)
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
                

                output_idx = torch.argmax(action_dist, 1).unsqueeze(1)
                action_idx = torch.tensor(self.category_action_offset[j]).cuda()[output_idx]
                
                is_end = (action_idx == self.end_idx).view(-1).float()
                
                #greedy_list.append(action_idx)
                greedy_list_category.append(action_idx)
                greedy_list.append(action_idx)
                
                self.update_lstm(action_idx, j)
                
                #money = money - self.prices[action_idx]
                #res_action_capacity[action_idx] -= 1
                #res_type_capacity[self.id2type[action_idx]] -= 1
                money = money - self.prices[action_idx.view(-1)]
                res_action_capacity -= torch.eye(res_action_capacity.size()[1]).cuda()[action_idx.view(-1)]
                res_type_capacity -= torch.eye(res_type_capacity.size()[1]).cuda()[torch.tensor(self.id2type).cuda()[action_idx.view(-1)]]
                
                #assert money >= 0
                #assert res_action_capacity[action_idx] >= 0
                #assert res_type_capacity[self.id2type[action_idx]] >= 0
                assert torch.sum((money >= 0).float()) == batch_size
                assert torch.sum((res_action_capacity >= 0).float()) == len(res_action_capacity.view(-1))
                assert torch.sum((res_type_capacity >= 0).float()) == len(res_type_capacity.view(-1))
            greedy_list_by_category.append(utils.remove_token(torch.cat(greedy_list_category, 1).cpu().numpy().tolist(), self.end_idx))
        greedy_list = torch.cat(greedy_list, 1).cpu().numpy().tolist()
        greedy_list = utils.remove_token(greedy_list, self.end_idx)
        
#         greedy_time = time.time()
#         print("greedy time:", greedy_time - sample_time)
        
        '''for j in range(self.output_categories):
            if gate and no_output_bi[j]:
                greedy_list_by_category.append([])
                continue
            else:
                # initialize LSTM
                self.initialize_lstm(init_embedding, (init_h, init_c), j)
                
                greedy_list_category = []
                for i in range(self.max_output_num):
                    H = self.history[-1][0][-1, :, :]
                    action_dist = self.classif_LN(H, j)
                    
                    action_mask = self.money_mask(money) * self.side_mask[side[0]]
                    #action_mask *= self.category_mask[j]
                    action_mask *= self.get_capacity_mask(torch.tensor(res_action_capacity).unsqueeze(0).cuda(), self.mute_action_mask).view(-1)
                    action_mask *= self.get_capacity_mask(torch.tensor(res_type_capacity).unsqueeze(0).cuda(), self.mute_type_mask).view(-1)
                    action_mask = action_mask[self.category_action_offset[j]]
                    action_dist = action_dist * action_mask # + (1 - action_mask) * EPSILON
                    
                    # nan in action_dist
                    #action_dist[action_dist != action_dist] = 0.0
                    
                    is_zero = (torch.sum(action_dist) == 0).float()
                    mask_all_tensor = torch.zeros_like(action_dist, dtype = torch.float).cuda().view(-1)
                    mask_all_tensor[-1] = 1.0
                    action_dist = action_dist + is_zero * mask_all_tensor
                    
                    output_idx = torch.argmax(action_dist).item()
                    action_idx = self.category_action_offset[j][output_idx]
                    if action_idx == self.end_idx:
                        break
                    #greedy_list.append(action_idx)
                    greedy_list_category.append(action_idx)
                    greedy_list.append(action_idx)
                    self.update_lstm(action_idx, j)
                    money = money - self.prices[action_idx]
                    res_action_capacity[action_idx] -= 1
                    res_type_capacity[self.id2type[action_idx]] -= 1
                    
                    assert money >= 0
                    assert res_action_capacity[action_idx] >= 0
                    assert res_type_capacity[self.id2type[action_idx]] >= 0
                greedy_list_by_category.append(greedy_list_category)'''
        
            
            
        
        return action_list, greedy_list, action_prob_by_category, no_output_bi.cpu().numpy().tolist(), bi_prob, action_list_by_category, greedy_list_by_category
            
        
    
    def initialize_lstm(self, init_embedding, init_state, category_id):
        if category_id == 0:
            self.history = [self.rnn1(init_embedding, init_state)[1]]
        elif category_id == 1:
            self.history = [self.rnn2(init_embedding, init_state)[1]]
        elif category_id == 2:
            self.history = [self.rnn3(init_embedding, init_state)[1]]
        else:
            raise NotImplementedError("Category ID exceeds number of output categories.")
    
    def update_lstm(self, action, category_id, offset=None):

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
        if category_id == 0:
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
#             print('category {}'.format(j))
            # initialize LSTM
            init_action = [self.start_idx] * batch_size * k
            init_embedding = self.get_embedding(init_action).unsqueeze(1)
            init_h_tile = utils.tile_along_beam(init_h, k, 1)
            init_c_tile = utils.tile_along_beam(init_c, k, 1)
            self.initialize_lstm(init_embedding, (init_h_tile, init_c_tile), j)
            # TODO: is_end tile: (batch_size,) - > (batch_size * last_k)
            is_end = utils.tile_along_beam(no_output_bi[:, j].float(), k, -1)
            
            
            for i in range(self.max_output_num):
#                 print(f'generate action {i}')
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
                # beiieb

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
        #self.LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        if self.shared_attention_weight:
            self.att_LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1 = nn.Linear(self.ff_dim, 1)
            self.att_LN2 = nn.Linear(self.embedding_dim + self.resource_dim, self.ff_dim)
            self.v2 = nn. Linear(self.ff_dim, 1)
        else:
            self.att_LN1_s = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_s = nn.Linear(self.ff_dim, 1)
            self.att_LN1_t = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_t = nn.Linear(self.ff_dim, 1)
            self.att_LN1_o = nn.Linear(self.embedding_dim, self.ff_dim)
            self.v1_o = nn.Linear(self.ff_dim, 1)
            
            self.att_LN2_t = nn.Linear(self.embedding_dim + self.resource_dim, self.ff_dim)
            self.v2_t = nn. Linear(self.ff_dim, 1)
            self.att_LN2_o = nn.Linear(self.embedding_dim + self.resource_dim, self.ff_dim)
            self.v2_o = nn. Linear(self.ff_dim, 1)
        
        self.HLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)
        self.CLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)
        
        self.LNDropout = nn.Dropout(p=self.ff_dropout_rate)
        
        assert self.output_categories == 3
        
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
        
        '''self.rnn = []
        self.BiClassif1 = []
        self.BiClassif2 = []
        self.BiClassif3 = []
        self.BiClassif4 = []
        for i in range(self.output_categories):
            
            self.BiClassif1.append(nn.Linear(self.input_dim, self.ff_dim))
            self.BiClassif2.append(nn.Linear(self.ff_dim, 2 * self.ff_dim))
            self.BiClassif3.append(nn.Linear(2 * self.ff_dim, self.ff_dim))
            self.BiClassif4.append(nn.Linear(self.ff_dim, 2))
            
            self.rnn.append(nn.LSTM(input_size = self.embedding_dim,
                           hidden_size = self.history_dim,
                           num_layers = self.history_num_layers,
                           batch_first = True))'''
        
        
        
        if torch.cuda.is_available():
            if self.shared_attention_weight:
                self.att_LN1 = self.att_LN1.cuda()
                self.v1 = self.v1.cuda()
                self.att_LN2 = self.att_LN2.cuda()
                self.v2 = self.v2.cuda()
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
            
            '''for i in range(self.output_categories):
                self.BiClassif1[i] = self.BiClassif1[i].cuda()
                self.BiClassif2[i] = self.BiClassif2[i].cuda()
                self.BiClassif3[i] = self.BiClassif3[i].cuda()
                self.BiClassif4[i] = self.BiClassif4[i].cuda()
                self.rnn[i] = self.rnn[i].cuda()'''
    
    def initialize_modules(self):
        # xavier initialization
        '''for i in range(self.output_categories):
            for name, param in self.rnn[i].named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)'''
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
        
        if self.shared_attention_weight:
            nn.init.xavier_uniform_(self.att_LN1.weight)
            nn.init.xavier_uniform_(self.att_LN2.weight)
            nn.init.xavier_uniform_(self.v1.weight)
            nn.init.xavier_uniform_(self.v2.weight)
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
        
        '''for i in range(self.output_categories):
            nn.init.xavier_uniform_(self.BiClassif1[i].weight)
            nn.init.xavier_uniform_(self.BiClassif2[i].weight)
            nn.init.xavier_uniform_(self.BiClassif3[i].weight)
            nn.init.xavier_uniform_(self.BiClassif4[i].weight)'''
    
    def clone(self, npy_dict):
        clone = CsgoModel(self.args, npy_dict)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

if __name__ == '__main__':
    def get_capacity_mask(res_capacity, mute_mask):
        # is_mute = (torch.tensor(res_action_capacity) == 0).float().cuda()
#         ret = []
#         for j in range(len(res_action_capacity)):
#             mask = torch.ones(mute_mask.size()[1]).cuda()
#             for i, res_cap in enumerate(res_action_capacity[j]):
#                 if res_cap == 0:
#                     mask *= mute_mask[i]
#             ret.append(mask.unsqueeze(0))
#         return torch.cat(ret, 0)
        to_multiply_mask = 1 - (res_capacity > 0).float()
        ret = (torch.mm(to_multiply_mask, mute_mask) == torch.sum(to_multiply_mask, 1).unsqueeze(1)).float()
        return ret
    mute_type_mask = torch.tensor([[0,0,1,1,1,1,1,1],[1,1,0,0,1,1,1,1],[1,1,1,1,0,0,1,1],[1,1,1,1,1,1,0,0]]).float()
    res_type_capacity = torch.randint(20,(3, 4)) - 10
    print("mask:")
    print(mute_type_mask)
    print("capacity:")
    print( (res_type_capacity>0).float())
    
    action_mask = get_capacity_mask(res_type_capacity, mute_type_mask)
    print(action_mask)
    assert 1 == 0
    
    
    # Parsing
    parser = argparse.ArgumentParser('Train MAML on CSGO')
    # params
    parser.add_argument('--logdir', default='log/', type=str, help='Folder to store everything/load')
    parser.add_argument('--player_mode', default='terrorist', type=str, help='terrorist or counter_terrorist')
    parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
    parser.add_argument('--start_meta_iteration', default=0, type=int, help='start number of meta iterations')
    parser.add_argument('--meta_iterations', default=100, type=int, help='number of meta iterations')
    parser.add_argument('--meta_lr', default=1., type=float, help='meta learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
    parser.add_argument('--check_every', default=1000, type=int, help='Checkpoint every')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')
    parser.add_argument('--action_embedding', default = '/home/derenlei/MAML/data/action_embedding.npy', help = 'Path to action embedding.')
    parser.add_argument('--action_name', default = '/home/derenlei/MAML/data/action_name.npy', help = 'Path to action name.')
    parser.add_argument('--action_money', default = '/home/derenlei/MAML/data/action_money.npy', help = 'Path to action money.')
    parser.add_argument('--side_mask', default = '/home/derenlei/MAML/data/mask.npz', help = 'Path to mask of two sides.')
    parser.add_argument('--history_dim', default = 512, help = 'LSTM hidden dimension.')
    parser.add_argument('--history_num_layers', default = 2, help = 'LSTM layer number.')
    parser.add_argument('--ff_dim', default = 256, help = 'MLP dimension.')
    parser.add_argument('--resource_dim', default = 2, help = 'Resource (money, performance, ...) dimension.')
    parser.add_argument('--ff_dropout_rate', default = 0.1, help = 'Dropout rate of MLP.')
    parser.add_argument('--max_output_num', default = 10, help = 'Maximum number of actions each round.')
    parser.add_argument('--beam_size', default = 128, help = 'Beam size of beam search predicting.')
    
    
    # args Processing
    args = parser.parse_args()
    print('start')
    model = CsgoModel(args)
    print('model created')
    # money = torch.randint(100, 1000, (10,))
    money_s = [torch.randint(100, 1000, (1,)).float().item()]
    money_t = torch.randint(100, 1000, (4,)).float()
    money_o = torch.randint(100, 1000, (5,)).float()
    money = (money_s, money_t, money_o)
    
    perf_s = [torch.randint(0, 5, (1,)).float().item()]
    perf_t = torch.randint(0, 5, (4,)).float()
    perf_o = torch.randint(0, 5, (5,)).float()
    perf = (perf_s, perf_t, perf_o)
    
    x_s = torch.randint(2, 20, (5,)).cuda()
    x_t = []
    for i in range(4):
        num_weapon = torch.randint(3, 10, (1,)).item()
        xt = torch.randint(2, 20, (num_weapon,)).cuda()
        x_t.append([xt, [money_t[i].item()], [perf_t[i].item()]])
    x_o = []
    for i in range(5):
        num_weapon = torch.randint(3, 10, (1,)).item()
        xo = torch.randint(2, 20, (num_weapon,)).cuda()
        x_o.append([xo, [money_o[i].item()], [perf_o[i].item()]])
    x = (x_s, x_t, x_o)
    
    side = [0]
    score = [0.8, 0.5]
    data = [side, x_s, money_s, perf_s, score, x_t, x_o]
    print('start_forward')
    action_list, action_prob, greedy_list = model.forward(data)
    print(action_list)
    print(action_prob)
    labels = torch.randint(2, 20, (torch.randint(3, 10, (1,)).item(),))
    labels[-1] = 1
    print(labels)
    print('loss')
    loss_dict = model.loss((action_list, action_prob, greedy_list), labels)
    print(loss_dict)
    print('prediction')
    action_list, action_prob = model.predict(data)
    print(action_list)
    print(action_prob)
