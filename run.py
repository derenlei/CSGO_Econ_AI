import torch
import random
import tqdm
import os
import argparse

#from MAML.args import argument_parser
from MAML.src.preprocess import read_dataset
from MAML.src.model import CsgoModel

from MAML.src.utils import find_latest_file
from MAML.src.utils import get_accuracy

from tensorboardX import SummaryWriter

DATA_DIR = 'data/'

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
        
        
def evaluation(model, optimizer, k_shot, val_set):
    losses = []
    accuracies = []
    model.eval()
    for i in range(len(val_set)):
        val_data_current = val_set[i]    
        for iteration in range(k_shot):
            # Sample minibatch
            data, labels = val_data_current[iteration]

            # Forward pass
            prediction = model.forward(data)

            # Get loss
            loss = model.loss(prediction, labels)
            
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
            loss = model.loss(prediction, labels)
            target_loss.append(loss)
            accuracy = get_accuracy(prediction, labels) # TODO
            target_acc.append(accuracy)
        
            # Get accuracy
            accuracy = get_accuracy(prediction, labels) # TODO

        losses.append(np.mean(target_loss))
        accuracies.append(np.mean(target_acc))

    return np.mean(losses), np.mean(accuracies)    
        
def insight_learning(model_insight, optimizer, k_shot, train_data_current):

    model_insight.train()
    
    # support set
    for iteration in range(k_shot):
        # Sample minibatch
        data, labels = train_data_current[iteration]
        # Forward pass
        # TODO:
        # x: [x_self, [x_teammate1, ...], [x_opponents1, ...]] weapon indexes of all players
        # money: [money_self, [money_teammate1, ...], [money_opponents1, ...]], normalized
        # performance: same as money
        
        prediction = model_insight.forward(data)

        # Get loss
        loss = model_insight.loss(prediction, labels)# TODO

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # target set
    target_loss = list()
    target_acc = list()
    for iteration in range(k_shot, len(train_data_current)):
        data, labels = train_data_current[iteration]
        prediction = model_insight.forward(data)
        loss = model_insight.loss(prediction, labels)
        target_loss.append(loss)
        accuracy = get_accuracy(prediction, labels) # TODO
        target_acc.append(accuracy)
        
    # Backward pass - Update fast net
    loss = np.mean(target_loss)
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
    parser.add_argument('--player_mode', default='terrorist', type=str, help='terrorist or counter_terrorist')
    parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
    parser.add_argument('--start_meta_iteration', default=0, type=int, help='start number of meta iterations')
    parser.add_argument('--meta_iterations', default=100, type=int, help='number of meta iterations')
    parser.add_argument('--meta_lr', default=1., type=float, help='meta learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
    parser.add_argument('--check_every', default=1000, type=int, help='Checkpoint every')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')
    
    # args Processing
    args = parser.parse_args()
    print(args)
    run_dir = args.logdir
    check_dir = os.path.join(run_dir, 'checkpoint')
    
    """
    Load data and construct model
    """
    random.seed(args.seed)
    train_set, val_set, test_set = read_dataset(DATA_DIR) # TODO: implement DATA_DIR
    # build model, optimizer
    model = CsgoModel() # TODO: add args
    if args.cuda:
        model.cuda()
    meta_optimizer = torch.optim.SGD(model.parameters(), lr=args.meta_lr) # TODO: add args
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
        model.load_state_dict(checkpoint['model'])
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
    logger = SummaryWriter(run_dir)
    
    early_stopping_counter = 0
    prev_val_acc = 0
    
    # Meta Train
    for meta_iteration in tqdm(args.start_meta_iteration, args.meta_iterations):
        
        train_data_current = random.choice(train_set)
        
        # Update learning rate
        meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
        learning_rate_decay(meta_optimizer, meta_lr)
    
        # Clone model
        model_insight = model.clone()
        optimizer = get_optimizer(model_insight, state)

        # Update insight model
        loss, accuracy = insight_learning(model_insight, optimizer, args.iterations, train_data_current) # TODO
        state = optimizer.state_dict()  # save optimizer state

        # Update slow net
        model.point_grad_to(model_insight)
        meta_optimizer.step()
        
        # save log
        info.setdefault(loss_eval, {})
        info.setdefault(accuracy_eval, {})
        info.setdefault(meta_lr, {})
        info['loss'][meta_iteration] = loss
        info['accuracy'][meta_iteration] = accuracy
        info['meta_lr'][meta_iteration] = meta_lr
        logger.add_scalar('loss', meta_loss, meta_iteration)
        logger.add_scalar('accuracy', meta_accuracy, meta_iteration)
        logger.add_scalar('meta_lr', meta_lr, meta_iteration)            
   
        # Meta Evaluation
        if meta_iteration % args.validate_every == 0:
            print('\n\nMeta-iteration', meta_iteration)
            print('(started at {})'.format(args.start_meta_iteration))
            print('Meta LR', meta_lr)
            
            # TODO, same as train, retrieve loss and accuracy
            # Clone model
            model_insight = model.clone()
            optimizer = get_optimizer(model_insight, state)

            # Update insight model
            eval_loss, eval_accuracy = evaluation(model_insight, optimizer, args.iterations, val_set) # TODO
            state = optimizer.state_dict()  # save optimizer state            
            
            # save log
            print('\n')
            print('average eval_loss', eval_loss)
            print('average eval_accuracy', eval_accuracy)
            logger.add_scalar('val_loss', eval_loss, meta_iteration)
            logger.add_scalar('val_accuracy', eval_accuracy, meta_iteration)         
            
        if meta_iteration % args.check_every == 0 and not (args.checkpoint and meta_iteration == args.start_meta_iteration):
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
            
        # Early stopping
        if prev_val_acc >= acc:
            early_stopping_counter += 1
        else:
            prev_val_acc = acc
            early_stopping_counter = 0
        if early_stopping_counter > 10:
            print('Validation performance not improving. Early stop.')
            break
        
    
    # Meta Test
    print('\nTest Performance')
    model_insight = model.clone()
    optimizer = get_optimizer(model_insight, state)

    # Update insight model
    test_loss, test_accuracy = evaluation(model_insight, optimizer, args.iterations, test_set) # TODO  
    print('average test_loss', test_loss)
    print('average test_accuracy', test_accuracy) 
    
if __name__ == '__main__':
    main()
    