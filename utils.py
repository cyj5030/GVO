import argparse
import yaml

def prepare_parser():
    parser = argparse.ArgumentParser(description='Parser for all scripts.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-config', type=str, default='./config/kitti.yaml',  help='yaml config file.')
    return parser

def ConfigFromFile(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = config['default']['model']

    outputs = {}
    outputs.update(config['default'])
    outputs.update(config[model])
    
    return outputs