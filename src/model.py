import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import Counter


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
    def __init__(self):
        super(CsgoModel, self).__init__()
        # ReptileModel.__init__(self)
        
        self.history_dim = 512
        self.history_num_layers = 2
        self.embedding_dim = 100
        self.ff_dim = 100
        self.input_dim = self.ff_dim * 3 + 6
        self.output_dim = 20
        self.ff_dropout_rate = 0.1
        self.rnn_dropout_rate = 0.1
        
        self.embedding = torch.tensor(np.load('/home/derenlei/MAML/data/action_embedding.npy')).cuda()
        self.id2name = np.load('/home/derenlei/MAML/data/action_money.npy', allow_pickle = True)
        self.id2money = np.load('/home/derenlei/MAML/data/action_name.npy', allow_pickle = True)
        self.id2money.dtype = int
        print(self.id2money.dtype)
        self.prices = torch.tensor(self.id2money).cuda()
        
        self.end_idx = self.embedding.size()[0] - 1
        self.start_idx = self.end_idx
        print(self.embedding[self.start_idx])
        self.action_mask = torch.ones(self.output_dim).float().cuda()
        self.action_mask[self.start_idx] = 0.0
        self.max_output_num = 10
        self.beam_size = 128
        assert 1 == 0

        #xavier_initialization
        
        self.define_modules()
        self.initialize_modules()
        
    def reward_fun(self, a, a_r):
        # F1 score
        a_common = list((Counter(a) & Counter(a_r.cpu().numpy())).elements())
        recall = len(a_common) / len(a_r)
        precision = len(a_common) / len(a)
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
        if not isinstance(idx, list):
            return self.embedding[idx]
        else:
            ret = []
            for i in idx:
                ret.append(self.embedding[i])
            return ret
    
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
    
    def forward(self, x, money, performance):
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        
        x_s, x_t, x_o = x
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
        hs = torch.cat([hs, torch.tensor([perf_s]).cuda(), torch.tensor([money_s]).cuda()], -1)
        
        # concat representations
        h = torch.cat([hs, ht, ho], -1)
        
        # return values - predictions and probabilities
        action_list = []
        greedy_list = []
        action_prob = []
        
        # initialize lstm
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        
        # generate predictions
        for i in range(self.max_output_num):
            H = self.history[-1][0][-1, :, :]
            action_dist = self.classif_LN(H)
            # action_mask = self.money_mask(money, self.prices) * self.action_mask
            # is_zero = (torch.sum(r_prob_b, 1) == 0).float().unsqueeze(1)
            action_mask = self.action_mask
            action_dist = action_dist * action_mask + (1 - action_mask) * EPSILON
            action_idx = torch.multinomial(action_dist, 1, replacement=True).item()
            action_list.append(action_idx)
            greedy_idx = torch.argmax(action_dist).item()
            greedy_list.append(greedy_idx)
            action_prob.append(action_dist[0][action_idx])
            self.update_lstm(action_idx)
            # TODO
            # money = money - self.prices[action_idx]
            
            if action_idx == self.end_idx:
                return action_list, action_prob, greedy_list
            # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
        
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
    
    
    def predict(self, x, money, performance):
        '''
        x: input
        '''
        '''
        forward for one round
        x: idx of weapons, x = [x_s, x_t, x_o]
        x_s: num_weapon #(num_batch, num_shot, num_weapon)
        x_t, x_o: (num_player, num_weapon) #(num_batch, num_shot, num_player, num_weapon)
        '''
        
        x_s, x_t, x_o = x
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
        hs = torch.cat([hs, torch.tensor([perf_s]).cuda(), torch.tensor([money_s]).cuda()], -1)
        
        # concat representations
        h = torch.cat([hs, ht, ho], -1)
        
        # return values - predictions and probabilities
        action_list = []
        action_prob = []
        
        # initialize lstm
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        
        log_action_prob = torch.zeros(1).cuda()
        finished = torch.zeros(1).cuda()
        
        action_list = torch.tensor([self.start_idx]).unsqueeze(0).cuda()
        money_s = torch.tensor([money_s]).cuda()
        
        # generate predictions
        for i in range(self.max_output_num):
            H = self.history[-1][0][-1, :, :]
            action_dist = self.classif_LN(H)
            action_mask = self.money_mask(money_s.view(-1, 1)) * self.action_mask
            # is_zero = (torch.sum(r_prob_b, 1) == 0).float().unsqueeze(1)
            action_mask = self.action_mask
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
            print(action_list[0])
            
            self.update_lstm(action_idx, offset = action_offset)
            # TODO
            money_s = money_s[action_offset].view(k)
            money_s = money_s - self.prices[action_idx].view(-1)
            
            finished = (action_idx == self.end_idx).float()
            if action_idx.view(-1)[0].item() == self.end_idx:
                break
            # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
        
        pred = action_list[0][1:]
        print(action_list)
        print(pred)
        actions = []
        for elem in pred:
            actions.append(elem)
            if elem == self.end_idx:
                break
        return actions, log_action_prob
        
    
    def define_modules(self):
        # terrorist
        #self.LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.att_LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.v1 = nn.Linear(self.ff_dim, 1)
        self.att_LN2 = nn.Linear(self.ff_dim + 2, self.ff_dim + 2)
        self.v2 = nn. Linear(self.ff_dim + 2, 1)
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

if __name__ == '__main__':
    print('start')
    model = CsgoModel()
    print('model created')
    # money = torch.randint(100, 1000, (10,))
    money_s = torch.randint(100, 1000, (1,)).float().item()
    money_t = torch.randint(100, 1000, (4,)).float()
    money_o = torch.randint(100, 1000, (5,)).float()
    money = (money_s, money_t, money_o)
    
    perf_s = torch.randint(0, 5, (1,)).float().item()
    perf_t = torch.randint(0, 5, (4,)).float()
    perf_o = torch.randint(0, 5, (5,)).float()
    perf = (perf_s, perf_t, perf_o)
    
    x_s = torch.randint(2, 20, (5,)).cuda()
    x_t = []
    for i in range(4):
        num_weapon = torch.randint(3, 10, (1,)).item()
        xt = torch.randint(2, 20, (num_weapon,)).cuda()
        x_t.append(xt)
    x_o = []
    for i in range(5):
        num_weapon = torch.randint(3, 10, (1,)).item()
        xo = torch.randint(2, 20, (num_weapon,)).cuda()
        x_o.append(xo)
    x = (x_s, x_t, x_o)
    print('start_forward')
    action_list, action_prob, greedy_list = model.forward(x, money, perf)
    print(action_list)
    print(action_prob)
    labels = torch.randint(2, 20, (torch.randint(3, 10, (1,)).item(),))
    labels[-1] = 1
    print(labels)
    print('loss')
    loss_dict = model.loss((action_list, action_prob, greedy_list), labels)
    print(loss_dict)
    print('prediction')
    action_list, action_prob = model.predict(x, money, perf)
    print(action_list)
    print(action_prob)