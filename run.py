import torch
import random
from tqdm import tqdm
import os
import argparse
import numpy as np

#from MAML.args import argument_parser
from src.preprocess import read_dataset
from src.model import CsgoModel

from src.utils import find_latest_file
from src.utils import get_accuracy

from tensorboardX import SummaryWriter

DATA_DIR = './data/0-2999.npy'

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def learning_rate_decay(meta_optimizer, meta_lr):
    for param_group in meta_optimizer.param_groups:
        param_group['lr'] = meta_lr  
        
def get_optimizer(args, model, state=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer

def evaluation(model, optimizer, k_shot, val_set):
    losses = []
    accuracies = []
    model.eval()
    for i in tqdm(range(len(val_set))):
        val_data_current = val_set[i]
        if len(val_data_current) <= 5:
            print('found data size less than 5')
            continue
        for iteration in range(k_shot):
            # Sample minibatch
            data, labels = val_data_current[iteration]

            # Forward pass
            prediction = model.forward(data)

            # Get loss
            loss_dict = model.loss(prediction, labels)
            loss = loss_dict['model_loss']
            
            # Backward pass - Update fast net
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # target set
        target_loss = list()
        target_acc = list()
        for iteration in range(k_shot, len(val_data_current)):
            data, labels = val_data_current[iteration]
            prediction = model.forward(data)
            loss_dict = model.loss(prediction, labels)
            loss = loss_dict['model_loss']
            target_loss.append(loss.unsqueeze(0).detach())
            accuracy = get_accuracy(prediction[0], labels) # TODO
            target_acc.append(accuracy)

        target_loss = torch.cat(target_loss, -1)
        losses.append(torch.mean(target_loss).unsqueeze(0))
        accuracies.append(np.mean(target_acc))

    losses = torch.cat(losses, -1)
    return torch.mean(losses), np.mean(accuracies)    
        
def insight_learning(model_insight, optimizer, k_shot, train_data_current):

    model_insight.train()
    
    # support set
    for iteration in range(k_shot):
        #print('few_shot iteration:', iteration)
        # Sample minibatch
        data, labels = train_data_current[iteration]
        # Forward pass
        # TODO:
        # x: [x_self, [x_teammate1, ...], [x_opponents1, ...]] weapon indexes of all players
        # money: [money_self, [money_teammate1, ...], [money_opponents1, ...]], normalized
        # performance: same as money
        
        prediction = model_insight.forward(data)

        # Get loss
        loss_dict = model_insight.loss(prediction, labels)# TODO
        loss = loss_dict['model_loss']

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # target set
    target_loss = list()
    target_acc = list()
    for iteration in range(k_shot, len(train_data_current)):
        #print('target_set iteration:', iteration)
        data, labels = train_data_current[iteration]
        prediction = model_insight.forward(data)
        loss_dict = model_insight.loss(prediction, labels)
        loss = loss_dict['model_loss']
        target_loss.append(loss.unsqueeze(0))
        accuracy = get_accuracy(prediction[0], labels) # TODO
        target_acc.append(accuracy)
        
    # Backward pass - Update fast net
    target_loss = torch.cat(target_loss, -1)
    loss = torch.mean(target_loss)
    accuracy = np.mean(target_acc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()        
    
    return loss, accuracy        
        
def main():
    
    """
    Load args
    """
    # Parsing
    parser = argparse.ArgumentParser('Train MAML on CSGO')
    # params
    parser.add_argument('--logdir', default='log/', type=str, help='Folder to store everything/load')
    parser.add_argument('--statedir', default='5shot', type=str, help='Folder name to store model state')
    parser.add_argument('--player_mode', default='terrorist', type=str, help='terrorist or counter_terrorist')
    parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
    parser.add_argument('--start_meta_iteration', default=0, type=int, help='start number of meta iterations')
    parser.add_argument('--meta_iterations', default=120000, type=int, help='number of meta iterations')
    parser.add_argument('--meta_lr', default=0.01, type=float, help='meta learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--validate_every', default=30000, type=int, help='validate every')
    parser.add_argument('--check_every', default=1000, type=int, help='Checkpoint every')
    parser.add_argument('--checkpoint', default='log/checkpoint', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')
    parser.add_argument('--action_embedding', default = '/home/derenlei/MAML/data/action_embedding.npy', help = 'Path to action embedding.')
    parser.add_argument('--action_name', default = '/home/derenlei/MAML/data/action_name.npy', help = 'Path to action name.')
    parser.add_argument('--action_money', default = '/home/derenlei/MAML/data/action_money.npy', help = 'Path to action money.')
    parser.add_argument('--money_scaling', default =1000, help = 'Scaling factor between money features and actual money.')
    parser.add_argument('--side_mask', default = '/home/derenlei/MAML/data/mask.npz', help = 'Path to mask of two sides.')
    parser.add_argument('--history_dim', default = 512, help = 'LSTM hidden dimension.')
    parser.add_argument('--history_num_layers', default = 2, help = 'LSTM layer number.')
    parser.add_argument('--ff_dim', default = 256, help = 'MLP dimension.')
    parser.add_argument('--resource_dim', default = 2, help = 'Resource (money, performance, ...) dimension.')
    parser.add_argument('--ff_dropout_rate', default = 0.1, help = 'Dropout rate of MLP.')
    parser.add_argument('--max_output_num', default = 10, help = 'Maximum number of actions each round.')
    parser.add_argument('--beam_size', default = 128, help = 'Beam size of beam search predicting.')
    parser.add_argument('--seed', default = 4164, help = 'random seed.')
    
    
    # args Processing
    args = parser.parse_args()
    print(args)
    #run_dir = args.logdir
    check_dir = args.logdir + '/checkpoint/' + args.statedir # os.path.join(run_dir, 'checkpoint')
    
    """
    Load data and construct model
    """
    random.seed(args.seed)
    train_set, val_set, test_set = read_dataset(DATA_DIR) # TODO: implement DATA_DIR
    # build model, optimizer
    model = CsgoModel(args) # TODO: add args
    model.print_all_model_parameters()
    if torch.cuda.is_available():
        model.cuda()
    meta_optimizer = torch.optim.SGD(model.parameters(), lr=args.meta_lr) # TODO: add args
    info = {}
    state = None
    
    """
    Load checkpoint
    """
    # checkpoint is directory -> Find last model or '' if does not exist
    if os.path.isdir(args.checkpoint): # TODO: add args
        latest_checkpoint = find_latest_file(check_dir)
        if latest_checkpoint:
            print('Latest checkpoint found:', latest_checkpoint)
            args.checkpoint = os.path.join(check_dir, latest_checkpoint)
    else:
        args.checkpoint = ''
    # Start fresh
    if args.checkpoint == '':
        print('No checkpoint. Starting fresh')
    # Load Checkpoint
    elif os.path.isfile(args.checkpoint):
        print('Attempting to load checkpoint', args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['meta_net'])
        meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        state = checkpoint['optimizer']
        args.start_meta_iteration = checkpoint['meta_iteration']
        info = checkpoint['info']
    else:
        raise ArgumentError('Bad checkpoint. Delete logdir folder to start over.')
    

    """
    Meta learner loop
    """
    # Create tensorboard logger
    logger = SummaryWriter(args.logdir + '/board/' + args.statedir)
    
    early_stopping_counter = 0
    prev_val_acc = 0
    
    # Meta Train
    train_loss, train_accuracy = [], []
    for meta_iteration in tqdm(range(args.start_meta_iteration, args.meta_iterations)):
        
        #print('meta_iteration: ', meta_iteration)
        train_data_current = random.choice(train_set)
        while len(train_data_current) <= 5:
            train_data_current = random.choice(train_set)
        
        # Update learning rate
        meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
        learning_rate_decay(meta_optimizer, meta_lr)
    
        # Clone model
        model_insight = model.clone()
        optimizer = get_optimizer(args, model_insight, state)

        # Update insight model
        loss, accuracy = insight_learning(model_insight, optimizer, args.shots, train_data_current) # TODO
        state = optimizer.state_dict()  # save optimizer state

        # Update slow net
        model.point_grad_to(model_insight)
        meta_optimizer.step()
        
        # calculate average
        loss = loss.detach().item()
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        
        # save log
        # info.setdefault('loss', {})
        # info.setdefault('accuracy', {})
        # info.setdefault('meta_lr', {})
        # info['loss'][meta_iteration] = loss
        # info['accuracy'][meta_iteration] = accuracy
        # info['meta_lr'][meta_iteration] = meta_lr
        
        if meta_iteration % 100 == 0:
            logger.add_scalar('loss', sum(loss)/(len(loss)*1.0), meta_iteration)
            logger.add_scalar('accuracy', sum(accuracy)/(len(accuracy)*1.0), meta_iteration)
            logger.add_scalar('meta_lr', meta_lr, meta_iteration)
            train_loss, train_accuracy = [], []
   
        # Meta Evaluation
        if meta_iteration % args.validate_every == 0 and meta_iteration > 0:
            print('Start evaluation')
            #print('\n\nMeta-iteration', meta_iteration)
            #print('(started at {})'.format(args.start_meta_iteration))
            #print('Meta LR', meta_lr)
            
            # TODO, same as train, retrieve loss and accuracy
            # Clone model
            model_insight = model.clone()
            optimizer = get_optimizer(args, model_insight, state)

            # Update insight model
            eval_loss, eval_accuracy = evaluation(model_insight, optimizer, args.shots, val_set) # TODO
            state = optimizer.state_dict()  # save optimizer state            
            
            # save log
            print('\n')
            print('average eval_loss', eval_loss)
            print('average eval_accuracy', eval_accuracy)
            logger.add_scalar('val_loss', eval_loss, meta_iteration)
            logger.add_scalar('val_accuracy', eval_accuracy, meta_iteration)  
            
            # Early stopping
            if prev_val_acc >= eval_accuracy:
                early_stopping_counter += 1
            else:
                prev_val_acc = eval_accuracy
                early_stopping_counter = 0
            if early_stopping_counter > 10:
                print('Validation performance not improving. Early stop.')
                break
            
        if meta_iteration % args.check_every == 0 and not (args.checkpoint and meta_iteration == args.start_meta_iteration):

            info.setdefault('loss', {})
            info.setdefault('accuracy', {})
            info.setdefault('meta_lr', {})
            info['loss'][meta_iteration] = loss.detach().item()
            info['accuracy'][meta_iteration] = accuracy
            info['meta_lr'][meta_iteration] = meta_lr
            
            # Make a checkpoint
            checkpoint = {
                'meta_net': model.state_dict(),
                'meta_optimizer': meta_optimizer.state_dict(),
                'optimizer': state,
                'meta_iteration': meta_iteration,
                'info': info
            }
            checkpoint_path = os.path.join(check_dir, 'check-{}.pth'.format(meta_iteration))
            torch.save(checkpoint, checkpoint_path)
            print('Saved checkpoint to', checkpoint_path)
            
            
        
    
    # Meta Test
    print('\nTest Performance')
    model_insight = model.clone()
    optimizer = get_optimizer(args, model_insight, state)

    # Update insight model
    test_loss, test_accuracy = evaluation(model_insight, optimizer, args.shots, test_set) # TODO  
    print('average test_loss', test_loss)
    print('average test_accuracy', test_accuracy) 
    
if __name__ == '__main__':
    main()
    
