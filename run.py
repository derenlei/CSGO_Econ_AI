import torch
import random
import tqdm
import os
import argparse

#from MAML.args import argument_parser
from MAML.src.preprocess import read_dataset
from MAML.src.model import CsgoModel

from MAML.src.utils import find_latest_file

from tensorboardX import SummaryWriter

DATA_DIR = 'data/'

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def insight_learning(model_insight, optimizer, iterations, TODO):

    model_insight.train()
    for iteration in xrange(iterations):
        # Sample minibatch
        data, labels = # TODO

        # Forward pass
        prediction = # TODO

        # Get loss
        loss = # TODO

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.data[0]        
        
def main():
    
    """
    Load args
    """
    # Parsing
    parser = argparse.ArgumentParser('Train MAML on CSGO')
    # params
    parser.add_argument('logdir', help='Folder to store everything/load')
    parser.add_argument('--meta-iterations', default=100, type=int, help='number of meta iterations')
    parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
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
    meta_oprimizer = torch.optim.SGD(model.parameters(), lr=args.meta_lr) # TODO: add args
    info = {}
    state = None
    
    """
    Load checkpoint
    """
    # checkpoint is directory -> Find last model or '' if does not exist
    if os.path.isdir(args.checkpoint): # TODO: add args
        latest_checkpoint = find_latest_file(check_dir)
        if latest_checkpoint:
            print 'Latest checkpoint found:', latest_checkpoint
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
    # Meta Train
    for meta_iteration in tqdm(args.start_meta_iteration, args.meta_iterations):
        # Update learning rate
        meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
        learning_rate_decay(meta_optimizer, meta_lr)
    
        # Clone model
        model_insight = model.clone()
        optimizer = get_optimizer(model_insight, state)

        # Update insight model
        loss = insight_learning(model_insight, optimizer, args.iterations, TODO) # TODO
        state = optimizer.state_dict()  # save optimizer state

        # Update slow net
        model.point_grad_to(model_insight)
        meta_optimizer.step()
        
        # Meta Evaluation
        if meta_iteration % args.validate_every == 0:
            print('\n\nMeta-iteration', meta_iteration)
            print('(started at {})'.format(args.start_meta_iteration))
            print('Meta LR', meta_lr)
            
            # TODO, same as train, retrieve loss and accuracy
            
            
            # save log
            info.setdefault(loss_eval, {})
            info.setdefault(accuracy_eval, {})
            info.setdefault(meta_lr, {})
            info['loss'][meta_iteration] = meta_loss
            info['accuracy'][meta_iteration] = meta_accuracy
            info['meta_lr'][meta_iteration] = meta_lr
            print('\n')
            print('average metaloss', np.mean(info[loss_eval].values()))
            print('average accuracy', np.mean(info[accuracy_eval].values()))
            logger.add_scalar('loss', meta_loss, meta_iteration)
            logger.add_scalar('accuracy', meta_accuracy, meta_iteration)
            logger.add_scalar('meta_lr', meta_lr, meta_iteration)            
            
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
            
if __name__ == '__main__':
    main()
    