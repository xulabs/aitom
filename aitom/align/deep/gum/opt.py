import argparse

parser = argparse.ArgumentParser(description='Gum-Net Demo Options')
parser.add_argument('--build_model', default=False, 
                    help='whether to use pretrained model or build a new model', action='store_true')
parser.add_argument('--initial_lr', default=1e-7, type=float,
                    help='initial learning rate')
parser.add_argument('--data_path', default='gum_demo_data.pickle', type=str,
                    help='path to demo data')
parser.add_argument('--model_path', default='gum_demo_model.h5', type=str,
                    help='path to pretrained model')

opt = parser.parse_args()
