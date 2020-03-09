import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import Counter


import argparse


EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31


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
    def __init__(self, args):
        super(CsgoModel, self).__init__()
        # ReptileModel.__init__(self)
        self.args = args
        self.embedding = torch.tensor(np.load(args.action_embedding)).cuda().float()
        self.id2name = list(np.load(args.action_name, allow_pickle = True))
        self.id2money = list(np.load(args.action_money, allow_pickle = True))
        self.prices = torch.tensor(self.id2money).cuda()
        self.money_scaling = args.money_scaling
        self.end_idx = self.embedding.size()[0] - 1
        self.start_idx = self.end_idx
        self.embedding_dim = self.embedding.size()[1]
        self.output_dim = self.embedding.size()[0]
        self.action_mask = torch.ones(self.output_dim).float().cuda()
        # self.action_mask[self.start_idx] = 0.0
        side_mask = np.load(args.side_mask)
        self.side_mask = dict()
        self.side_mask[0] = torch.tensor(side_mask['t_mask'].astype(float)).cuda()
        self.side_mask[1] = torch.tensor(side_mask['ct_mask'].astype(float)).cuda()
        self.side_embedding = torch.tensor([[1, 0], [0, 1]]).cuda().float()
        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.ff_dim = args.ff_dim
        self.resource_dim = args.resource_dim
        self.team_dim = 4
        self.input_dim = (self.embedding_dim + self.resource_dim) * 3 + self.team_dim
        
        self.ff_dropout_rate = args.ff_dropout_rate
        self.max_output_num = args.max_output_num
        self.beam_size = args.beam_size

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
        a_r_new = a_r[: -1]
        # both are empty
        if len(a_new) == 0 and len(a_r_new) == 0:
            return 1.0
        a_common = list((Counter(a_new) & Counter(a_r_new)).elements())
        recall = len(a_common) / (len(a_r) + EPSILON)
        precision = len(a_common) / (len(a) + EPSILON)
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

        # TODO: batch
        # x, money, gs_actions = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        
        action_list, action_prob, greedy_list = predictions
        log_action_probs = []
        for ap in action_prob:
            log_action_probs.append(torch.log(ap + EPSILON).unsqueeze(0))
        log_action_probs = torch.cat(log_action_probs, 0)
        # log_action_probs = torch.log(action_prob + EPSILON)
        gs_actions = labels

        # Compute policy gradient loss
        # Compute discounted reward
        final_reward = self.reward_fun(action_list, gs_actions) - self.reward_fun(greedy_list, gs_actions)
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
        pg_loss = -final_reward * torch.sum(log_action_probs)
        pt_loss = -final_reward * torch.sum(torch.exp(log_action_probs))
        
        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        
        loss_dict['reward'] = final_reward
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
    
    def high_att(self, x):
        # team-level attention
        h = self.att_LN2(x)
        h = torch.tanh(h)
        h = self.v2(h)
        att = F.softmax(h, 0)
        ret = torch.sum(att * x, 0)
        return ret
        
    def low_att(self, x):
        # player-level attention
        h = self.att_LN1(x)
        h = torch.tanh(h)
        h = self.v1(h)
        att = F.softmax(h, 0)
        ret = torch.sum(att * x, 0)
        return ret
    
    def classif_LN(self, x):
        out = self.LN1(x)
        out = F.relu(out)
        out = self.LNDropout(out)
        out = self.LN2(out)
        out = self.LNDropout(out)
        out = F.softmax(out, dim = -1)
        #action_dist = F.softmax(
        #    torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
        return out
    
    def money_mask(self, money):
        return (self.prices <= money).float().cuda() 
    
    def forward(self, data):
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        
        side, x_s, money_s, perf_s, score, x_t, x_o = data
        # represent allies
        ht = []
        for xti, moneyi, perfi in x_t:
            hti = self.low_att(self.get_embedding(xti))
            hti = torch.cat([hti, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
            ht.append(hti.unsqueeze(0))
        ht = torch.cat(ht, 0)
        ht = self.high_att(ht)
        
        # represent enemies
        ho = []
        for xoi, moneyi, perfi in x_o:
            hoi = self.low_att(self.get_embedding(xoi))
            hoi = torch.cat([hoi, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
            ho.append(hoi.unsqueeze(0))
        ho = torch.cat(ho, 0)
        ho = self.high_att(ho)
        
        # represent self
        hs = self.low_att(self.get_embedding(x_s))
        hs = torch.cat([hs, torch.tensor(money_s).cuda(), torch.tensor(perf_s).cuda()], -1)
        
        
        '''x_s, x_t, x_o = x
        perf_s, perf_t, perf_o = performance
        money_s, money_t, money_o = money
        x_s = self.get_embedding(x_s)
        x_t = self.get_embedding(x_t)
        x_o = self.get_embedding(x_o)
        
        # represent allies
        ht = []
        for xti, perfi, moneyi in zip(x_t, perf_t, money_t):
            hti = self.low_att(xti)
            hti = torch.cat([hti, torch.tensor([perfi]).cuda(), torch.tensor([moneyi]).cuda()], -1)
            ht.append(hti.unsqueeze(0))
        ht = torch.cat(ht, 0)
        ht = self.high_att(ht)
        
        # represent enemies
        ho = []
        for xoi, perfi, moneyi in zip(x_o, perf_o, money_o):
            hoi = self.low_att(xoi)
            hoi = torch.cat([hoi, torch.tensor([perfi]).cuda(), torch.tensor([moneyi]).cuda()], -1)
            ho.append(hoi.unsqueeze(0))
        ho = torch.cat(ho, 0)
        ho = self.high_att(ho)

        # represent self
        hs = self.low_att(x_s)
        hs = torch.cat([hs, torch.tensor([perf_s]).cuda(), torch.tensor([money_s]).cuda()], -1)'''
        
        # concat representations
        h = torch.cat([hs, ht, ho], -1)
        
        # incorporate team information
        h = torch.cat([h, self.side_embedding[side[0]], torch.tensor(score).cuda()], -1)
        
        # return values - predictions and probabilities
        action_list = []
        greedy_list = []
        action_prob = []
        money_s = money_s[0] * self.money_scaling
        
        # initialize lstm
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        
        # generate predictions
        for i in range(self.max_output_num):
            H = self.history[-1][0][-1, :, :]
            action_dist = self.classif_LN(H)
            action_mask = self.money_mask(money_s) * self.action_mask * self.side_mask[side[0]]
            # is_zero = (torch.sum(r_prob_b, 1) == 0).float().unsqueeze(1)
            action_dist = action_dist * action_mask + (1 - action_mask) * EPSILON
            action_idx = torch.multinomial(action_dist, 1, replacement=True).item()
            action_prob.append(action_dist[0][action_idx])
            if action_idx == self.end_idx:
                break
            action_list.append(action_idx)
            # greedy_idx = torch.argmax(action_dist).item()
            # greedy_list.append(greedy_idx)
            self.update_lstm(action_idx)
            # TODO
            # money = money - self.prices[action_idx]
            # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
        # greedy
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        for i in range(self.max_output_num):
            H = self.history[-1][0][-1, :, :]
            action_dist = self.classif_LN(H)
            action_mask = self.money_mask(money_s) * self.action_mask * self.side_mask[side[0]]
            # is_zero = (torch.sum(r_prob_b, 1) == 0).float().unsqueeze(1)
            action_dist = action_dist * action_mask + (1 - action_mask) * EPSILON
            action_idx = torch.argmax(action_dist).item()
            if action_idx == self.end_idx:
                break
            greedy_list.append(action_idx)
            self.update_lstm(action_idx)
            # TODO
            # money = money - self.prices[action_idx]
            
            
        
        '''print('ggg')
        print(greedy_list)
        print('sss')
        print(action_list)'''
        return action_list, action_prob, greedy_list
            
        
    
    def initialize_lstm(self, representation, init_action):
        init_embedding = self.get_embedding(init_action).unsqueeze(0).unsqueeze(1)
        # transform representation to initialize (h, c)
        init_h = self.HLN(representation).view(self.history_num_layers, 1, self.history_dim)
        init_c = self.CLN(representation).view(self.history_num_layers, 1, self.history_dim)
        self.history = [self.rnn(init_embedding, (init_h, init_c))[1]]
    
    def update_lstm(self, action, offset=None):

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]


        # update action history
        #if self.relation_only_in_path:
        #    action_embedding = kg.get_relation_embeddings(action[0])
        #else:
        #    action_embedding = self.get_action_embedding(action, kg)
        embedding = self.get_embedding(action).view(-1, 1, self.embedding_dim)
        if offset is not None:
            offset_path_history(self.history, offset.view(-1))
            # during inference, update batch size
            # self.hidden_tensor = offset_rule_history(self.hidden_tensor, offset)
            # self.cell_tensor = offset_rule_history(self.cell_tensor, offset)
            

        # self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])
        torch.backends.cudnn.enabled = False
        self.history.append(self.rnn(embedding, self.history[-1])[1])

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
    
    
    def predict(self, data):
        '''
        x: input
        '''
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        
        side, x_s, money_s, perf_s, score, x_t, x_o = data
        # represent allies
        ht = []
        for xti, moneyi, perfi in x_t:
            hti = self.low_att(self.get_embedding(xti))
            hti = torch.cat([hti, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
            ht.append(hti.unsqueeze(0))
        ht = torch.cat(ht, 0)
        ht = self.high_att(ht)
        
        # represent enemies
        ho = []
        for xoi, moneyi, perfi in x_o:
            hoi = self.low_att(self.get_embedding(xoi))
            hoi = torch.cat([hoi, torch.tensor(moneyi).cuda(), torch.tensor(perfi).cuda()], -1)
            ho.append(hoi.unsqueeze(0))
        ho = torch.cat(ho, 0)
        ho = self.high_att(ho)
        
        # represent self
        hs = self.low_att(self.get_embedding(x_s))
        hs = torch.cat([hs, torch.tensor(money_s).cuda(), torch.tensor(perf_s).cuda()], -1)
        
        '''x_s, x_t, x_o = x
        perf_s, perf_t, perf_o = performance
        money_s, money_t, money_o = money
        x_s = self.get_embedding(x_s)
        x_t = self.get_embedding(x_t)
        x_o = self.get_embedding(x_o)
        
        # represent allies
        ht = []
        for xti, perfi, moneyi in zip(x_t, perf_t, money_t):
            hti = self.low_att(xti)
            hti = torch.cat([hti, torch.tensor([perfi]).cuda(), torch.tensor([moneyi]).cuda()], -1)
            ht.append(hti.unsqueeze(0))
        ht = torch.cat(ht, 0)
        ht = self.high_att(ht)
        
        # represent enemies
        ho = []
        for xoi, perfi, moneyi in zip(x_o, perf_o, money_o):
            hoi = self.low_att(xoi)
            hoi = torch.cat([hoi, torch.tensor([perfi]).cuda(), torch.tensor([moneyi]).cuda()], -1)
            ho.append(hoi.unsqueeze(0))
        ho = torch.cat(ho, 0)
        ho = self.high_att(ho)

        # represent self
        hs = self.low_att(x_s)
        hs = torch.cat([hs, torch.tensor([perf_s]).cuda(), torch.tensor([money_s]).cuda()], -1)'''
        
        # concat representations
        h = torch.cat([hs, ht, ho], -1)
        
        # incorporate team information
        h = torch.cat([h, self.side_embedding[side[0]], torch.tensor(score).cuda()], -1)
        
        # return values - predictions and probabilities
        action_list = []
        action_prob = []
        
        # initialize lstm
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        
        log_action_prob = torch.zeros(1).cuda()
        finished = torch.zeros(1).cuda()
        
        action_list = torch.tensor([self.start_idx]).unsqueeze(0).cuda()
        money_s = torch.tensor([money_s]).cuda() * money_scaling
        
        # generate predictions
        for i in range(self.max_output_num):
            H = self.history[-1][0][-1, :, :]
            action_dist = self.classif_LN(H)
            action_mask = self.money_mask(money_s.view(-1, 1)) * self.action_mask * self.side_mask[side[0]]
            # is_zero = (torch.sum(r_prob_b, 1) == 0).float().unsqueeze(1)
            action_dist = action_dist * action_mask + (1 - action_mask) * EPSILON
            end_mask = torch.zeros(self.output_dim).cuda()
            end_mask[self.end_idx] = 1.0
            action_dist = action_dist * (1 - finished).view(-1, 1) + finished.view(-1, 1) * end_mask
            
            
            log_action_dist = log_action_prob.view(-1, 1) + torch.log(action_dist + EPSILON)
            assert log_action_dist.size()[1] == self.output_dim
            last_k = len(log_action_dist)
            log_action_dist = log_action_dist.view(1, -1)
            
            k = min(self.beam_size, log_action_dist.size()[1])
            log_action_prob, action_ind = torch.topk(log_action_dist, k)
            action_idx = action_ind % self.output_dim
            action_offset = action_ind / self.output_dim
            action_list = torch.cat([action_list[action_offset].view(k, -1), action_idx.view(-1, 1)], 1)
            #print(action_list[0])
            
            self.update_lstm(action_idx, offset = action_offset)
            # TODO
            money_s = money_s[action_offset].view(k)
            money_s = money_s - self.prices[action_idx].view(-1)
            
            finished = (action_idx == self.end_idx).float()
            if action_idx.view(-1)[0].item() == self.end_idx:
                break
            # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
        
        pred = action_list[0][1:]
        #print(action_list)
        #print(pred)
        actions = []
        for elem in pred:
            if elem == self.end_idx:
                break
            actions.append(elem)
        return actions, log_action_prob
        
    
    def define_modules(self):
        # terrorist
        #self.LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.att_LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.v1 = nn.Linear(self.ff_dim, 1)
        self.att_LN2 = nn.Linear(self.embedding_dim + self.resource_dim, self.ff_dim)
        self.v2 = nn. Linear(self.ff_dim, 1)
        self.HLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)
        self.CLN = nn.Linear(self.input_dim, self.history_dim * self.history_num_layers)
        self.LN1 = nn.Linear(self.history_dim, self.ff_dim)
        self.LN2 = nn.Linear(self.ff_dim, self.output_dim)
        self.LNDropout = nn.Dropout(p=self.ff_dropout_rate)
        self.rnn = nn.LSTM(input_size = self.embedding_dim,
                           hidden_size = self.history_dim,
                           num_layers = self.history_num_layers,
                           batch_first = True)
        
        if torch.cuda.is_available():
            self.att_LN1 = self.att_LN1.cuda()
            self.v1 = self.v1.cuda()
            self.att_LN2 = self.att_LN2.cuda()
            self.v2 = self.v2.cuda()
            self.LN1 = self.LN1.cuda()
            self.LN2 = self.LN2.cuda()
            self.HLN = self.HLN.cuda()
            self.CLN = self.CLN.cuda()
            self.LNDropout = self.LNDropout.cuda()
            self.rnn = self.rnn.cuda()
    
    def initialize_modules(self):
        # xavier initialization
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_uniform_(self.att_LN1.weight)
        nn.init.xavier_uniform_(self.att_LN2.weight)
        nn.init.xavier_uniform_(self.v1.weight)
        nn.init.xavier_uniform_(self.v2.weight)
        nn.init.xavier_uniform_(self.LN1.weight)
        nn.init.xavier_uniform_(self.LN2.weight)
        nn.init.xavier_uniform_(self.HLN.weight)
        nn.init.xavier_uniform_(self.CLN.weight)
    
    def clone(self):
        clone = CsgoModel(self.args)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

if __name__ == '__main__':
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