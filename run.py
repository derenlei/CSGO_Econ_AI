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
    for meta_iteration in tqdm(args.start_meta_iteration, args.meta_iterations):
        # Update learning rate
        meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
        learning_rate_decay(meta_optimizer, meta_lr)
        # TODO
    
if __name__ == '__main__':
    main()
    