import argparse

def parse_cli():
    parser = argparse.ArgumentParser(description="Hyperparameters of experiment")
    parser.add_argument('batch_size', default=128)
    parser.add_argument('exp_name', default=128)
    args = parser.parse_args()
    args_dict = vars(args)