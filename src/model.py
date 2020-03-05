import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

class csgoModel(reptileModel):
    def __init__(self, num_classes):
        super(csgoModel, self).__init__()
        # ReptileModel.__init__(self)
        
        self.history_dim = 512
        self.history_num_layers = 2
        self.embedding_dim = 100
        self.ff_dim = 100
        self.input_dim = self.ff_dim * 3
        self.output_dim = 20
        self.ff_dropout_rate = 0.1
        self.rnn_dropout_rate = 0.1
        
        self.prices = [] # TODO
        self.start_idx = 0
        self.end_idx = 1
        #self.action_mask = torch.ones(self.output_dim).float().cuda()
        #self.action_mask[self.start_idx] = 0.0
        self.max_output_num = 10
        
        self.embedding = None # TODO: load

        #xavier_initialization
        
        self.define_modules()
        self.initialize_modules()
        
    def reward_fun(self, a, a_r):
        # TODO: recall
        

    def loss(self, mini_batch):
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
        x, money, gs_actions = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        
        action_list, action_prob, greedy_list = self.forward(x, money)
        log_action_probs = torch.log(action_prob + EPSILON)

        # Compute policy gradient loss
        # Compute discounted reward
        final_reward = reward_fun(action_list, gs_actions) - reward_fun(greedy_action, gs_actions)
        if self.baseline != 'n/a':
            #print('stablized reward')
            final_reward = stablize_reward(final_reward)
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
        return self.embedding[idx]
    
    def encode(self, x):
        
        def hier_att(x_i):
            # lower-level attention
            h_i = []
            for xi in x_i:
                hi = self.att_LN1(xi)
                hi = F.tanh(hi)
                hi = self.v1(hi)
                att = F.softmax(hi, 0)
                hi = torch.sum(att * xi, 0)
                h_i.append(hi.unsqueeze(0))
            h_i = torch.cat(h_i, 0)
            
            # higher-level attention
            hi = self.att_LN2(h_i)
            hi = F.tanh(hi)
            hi = self.v2(hi)
            att = F.softmax(hi, 0)
            hi = torch.sum(att * h_i, 0)
            
            return hi
        
        
        
        # (self, teammate, opponent) = x
        x_s, x_t, x_o = x
        
        #hs = low_att(x_s)
        ht = hier_att(x_t)
        ho = hier_att(x_o) # TODO: diff weight for x_o attn
        return ht, ho
        '''h = torch.cat([hs, ht, ho], -1)
        
        out = self.LN1(h)
        out = F.relu(out)
        out = self.LNDropout(out)
        out = self.LN2(out)
        out = self.LNDropout(out)
        return out'''
    
        
    
    def forward(self, x, money):
        
        def low_att(x_i):
            hi = self.att_LN1(x_i)
            hi = F.tanh(hi)
            hi = self.v1(hi)
            att = F.softmax(hi, 0)
            hi = torch.sum(att * x_i, 0)
            return hi
        
        def classif_LN(xi):
            out = self.LN1(xi)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN2(out)
            out = self.LNDropout(out)
            out = F.softmax(out)
            #action_dist = F.softmax(
            #    torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            return out
        
        def money_mask(money, prices):
            return (torch.tensor(prices) <= money).float().cuda() 
        
        x_s, x_t, x_o = x
        ht, ho = self.encode(x)
        action_list = []
        greedy_action = []
        action_prob = []
        hs = low_att(x_s)
        h = torch.cat([hs, ht, ho], -1)
        init_action = self.start_idx
        self.initialize_lstm(h, init_action)
        for i in range(self.max_output_num):
            '''hs = low_att(xs)
            h = torch.cat([hs, ht, ho], -1)
            
            out = self.LN1(h)
            out = F.relu(out)
            out = self.LNDropout(out)
            out = self.LN2(out)
            out = F.softmax(out)'''
            
            H = self.history[-1][0][-1, :, :]
            action_dist = classif_LN(H)
            action_dist = action_dist * money_mask(money, self.prices)
            action_idx = torch.multinomial(action_dist, 1, replacement=True)
            action_list.append(action_idx)
            greedy_idx = torch.argmax(action_dist)
            greedy_list.append(greedy_idx)
            action_prob.append(action_dist[action_idx])
            self.update_lstm(action_idx)
            money = money - self.prices[action_idx]
            
            if action_idx == self.end_idx:
                return action_list, action_prob, greedy_list
            # xs = torch.cat([xs.unsqueeze(0), self.get_embedding(out_id).unsqueeze(0)], 0)
        
        return action_list, action_prob, greedy_list
            
        
    
    def initialize_lstm(self, h， init_action):
        init_embedding = self.get_embedding(init_action).unsqueeze(1)
        init_h = self.HLN(h).view(self.history_num_layers, 1, self.history_dim)
        init_h = self.CLN(h).view(self.history_num_layers, 1, self.history_dim)
        self.history = [self.rnn(init_embedding, (init_h, init_c))[1]]
    
    def update_lstm(self, action, offset=None):

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        def offset_rule_history(x, offset):
            return x[:, offset, :]

        # update action history
        #if self.relation_only_in_path:
        #    action_embedding = kg.get_relation_embeddings(action[0])
        #else:
        #    action_embedding = self.get_action_embedding(action, kg)
        embedding = self.get_embedding(action)
        if offset is not None:
            offset_path_history(self.history, offset)
            # during inference, update batch size
            # self.hidden_tensor = offset_rule_history(self.hidden_tensor, offset)
            # self.cell_tensor = offset_rule_history(self.cell_tensor, offset)
            

        # self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])
        torch.backends.cudnn.enabled = False
        self.history.append(self.rnn(embedding.unsqueeze(1), self.history[-1])[1])

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
    
    def run_train(self, train_data, dev_data):
        # TODO
        self.print_all_model_parameters()
        
        writer = SummaryWriter(self.board_dir)
        
        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        batch_num = 0
        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            # TODO omit this for faster speed
            # if self.rl_variation_tag.startswith('rs'):
            #     # Reward shaping module sanity check:
            #     #   Make sure the reward shaping module output value is in the correct range
            #     train_scores = self.test_fn(train_data)
            #     dev_scores = self.test_fn(dev_data)
            #     print('Train set average fact score: {}'.format(float(train_scores.mean())))
            #     print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            top_rules_hit = [] # percentage of hitting top rules
            if self.run_analysis:
                rewards = None
                fns = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                
                #import cProfile, pstats, io
                #from io import StringIO
                #import time
                #pr = cProfile.Profile()
                #pr.enable()  # start profiling
                    
                loss = self.loss(mini_batch)
                
                
                #s = io.StringIO()
                #sortby = 'cumulative'
                #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                #ps.print_stats()
                #print(s.getvalue())
                #assert(0==1)
                
                
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                
                self.optim.step()

                batch_losses.append(loss['print_loss'])
                
                if batch_num%10 == 0:
                    if 'top_rule_hit' in loss.keys():
                        writer.add_scalar('data/top_rule_hit', float(loss['top_rule_hit']), batch_num)
                    if 'reward' in loss.keys():
                        writer.add_scalar('data/reward', float(torch.mean(loss['reward'])), batch_num)
                batch_num += 1
                
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if 'top_rule_hit' in loss:
                    top_rules_hit.append(loss['top_rule_hit'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
            if entropies:
                stdout_msg += ' entropy = {}'.format(np.mean(entropies))
            if 'top_rule_hit' in loss:
                stdout_msg += ' top rules percentage = {}'.format(np.mean(top_rules_hit))
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
                self.eval()
                self.batch_size = self.dev_batch_size
                dev_scores, rule_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (correct evaluation)')
                hit1, hit3, hit5, hit10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                # tensorboard
                writer.add_scalar('data/hit1_dev_correct', hit1, epoch_id)
                writer.add_scalar('data/hit3_dev_correct', hit3, epoch_id)
                writer.add_scalar('data/hit5_dev_correct', hit5, epoch_id)
                writer.add_scalar('data/hit10_dev_correct', hit10, epoch_id)
                writer.add_scalar('data/mrr_dev_correct', mrr, epoch_id)
                if rule_scores is not None:
                    writer.add_scalar('data/confidence_score_dev', torch.mean(rule_scores).cpu().numpy(), epoch_id)
                
                # metrics = mrr
                metrics = hit1
                if self.kg.args.pre_train:
                    metrics = torch.mean(rule_scores).cpu().numpy()
                if rule_scores is not None:
                    print('average dev set rule confidence score: {}'.format(torch.mean(rule_scores).cpu().numpy()))
                print('Dev set performance: (include test set labels)')
                hit1, hit3, hit5, hit10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                # tensorboard
                writer.add_scalar('data/hit1_dev_test_label', hit1, epoch_id)
                writer.add_scalar('data/hit3_dev_test_label', hit3, epoch_id)
                writer.add_scalar('data/hit5_dev_test_label', hit5, epoch_id)
                writer.add_scalar('data/hit10_dev_test_label', hit10, epoch_id)
                writer.add_scalar('data/board/mrr_dev_test_label', mrr, epoch_id)
                
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(
                            dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
    
    def predict(self, prob):
        // TODO
    
    def define_modules(self):
        # terrorist
        #self.LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.att_LN1 = nn.Linear(self.embedding_dim, self.ff_dim)
        self.v1 = nn.Linear(self.ff_dim, 1)
        self.att_LN2 = nn.Linear(self.ff_dim, self.ff_dim)
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