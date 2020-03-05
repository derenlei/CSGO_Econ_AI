import torch
import random

from MAML.args import argument_parser
from MAML.preprocess import read_dataset

DATA_DIR = 'data/'

def main():
    """
    Load data and train model
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)
    train_set, val_set, test_set = read_dataset(DATA_DIR)
    
if __name__ == '__main__':
    main()
    