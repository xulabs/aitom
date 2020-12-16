from .train_double import *
from .data_gene import *
from .list_gene import *
from .train_double import *
from .test import *
import argparse


def main(args):
    # prepare data for training and testing.
    gene_train_and_test(args)
    gene_train_list(args)
    gene_verification_pair(args)
    # training.
    main_train(args)
    # testing
    test(args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data',
                        type=str,
                        help='Directory where raw data lies.',
                        default='./extracted_data.npz')
    parser.add_argument('--num_class_train',
                        type=int,
                        help='Number of class for training.',
                        default=17)
    parser.add_argument('--num_class_test',
                        type=int,
                        help='Number of class for testing.',
                        default=6)
    parser.add_argument('--num_class', type=int, help='Number of class.', default=23)
    parser.add_argument('--num_sample_per_class', type=int, help='Number of class.', default=50)
    parser.add_argument('--test_list_dir',
                        type=str,
                        help='test verification dir.',
                        default='./test.txt')
    parser.add_argument('--train_data_dir',
                        type=str,
                        help='Path to the train data directory.',
                        default='./train/')
    parser.add_argument('--test_data_dir',
                        type=str,
                        help='Path to the test data directory.',
                        default='./test/')
    parser.add_argument('--train_list_dir',
                        type=str,
                        help='Image list file for training',
                        default='./train.txt')
    parser.add_argument('--logs_base_dir',
                        type=str,
                        help='Directory where to write event logs.',
                        default='logs/demo_logs')
    parser.add_argument('--models_base_dir',
                        type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='models/demo_model')
    parser.add_argument('--images_per_class',
                        type=int,
                        help='Number of images per class.',
                        default=20)
    parser.add_argument('--class_per_batch',
                        type=int,
                        help='Number of classes per batch.',
                        default=5)
    parser.add_argument('--pretrained_model',
                        type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--epoch_size', type=int, help='Number of batches per epoch.', default=600)
    parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.', default=100)
    parser.add_argument('--image_height',
                        type=int,
                        help='Image size (height, width) in pixels.',
                        default=40)
    parser.add_argument('--image_size', type=int, help='Image size.', default=40)
    parser.add_argument('--image_width',
                        type=int,
                        help='Image size (height, width) in pixels.',
                        default=40)
    parser.add_argument('--center_loss_alfa',
                        type=float,
                        help='Center update rate for center loss.',
                        default=0.05)
    parser.add_argument('--learning_rate_decay_epochs',
                        type=int,
                        help='Number of epochs between learning rate decay.',
                        default=20)
    parser.add_argument('--learning_rate_decay_factor',
                        type=float,
                        help='Learning rate decay factor.',
                        default=0.9)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['ADAGRAD', 'ADAM', 'MOM', 'SGD'],
                        help='The optimization algorithm to use',
                        default='ADAM')
    parser.add_argument('--center_weighting',
                        type=float,
                        help='weights for balancing the center loss and other loss.',
                        default=1.0)
    parser.add_argument('--loss_type',
                        type=str,
                        choices=['softmax', 'lmcl', 'center', 'lmccl'],
                        help='Which type loss to be used.',
                        default='lmccl')
    parser.add_argument('--network',
                        type=str,
                        help='which network is used to extract feature.',
                        default='sphere_network')
    parser.add_argument('--gpu', type=str, help='which gpu to use.', default='2')
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=0.9)
    parser.add_argument('--alpha', type=float, help='Margin for cos margin.', default=0.35)
    parser.add_argument('--scale',
                        type=float,
                        help='Scale as the fixed norm of weight and feature.',
                        default=64.)
    parser.add_argument('--embedding_size',
                        type=int,
                        help='Dimensionality of the embedding.',
                        default=1024)
    parser.add_argument('--fc_bn', help='Wheater use bn after fc.', type=bool, default=True)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight regularization.',
                        default=0.0001)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=0.001)
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='Number of images to process in a batch in the test set.',
                        default=200)
    parser.add_argument(
        '--test_nrof_folds',
        type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.',
        default=10)
    parser.add_argument(
        '--model',
        type=str,
        default='',
        help='Could be either a directory containing the meta_file and ckpt_file for testing.')
    # parser.add_argument('--weight', type=float,
    #     help='weiht to balance the dist and th loss.', default=2.)
    # parser.add_argument('--model_def', type=str,
    #     help='Model definition. Points to a module containing the definition of the inference graph.',\
    #  default='models.inception_resnet_v1')
    # parser.add_argument('--batch_size', type=int,
    #     help='Number of images to process in a batch.', default=90)
    # parser.add_argument('--random_crop',
    #     help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
    #      'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    # parser.add_argument('--random_flip',
    #     help='Performs random horizontal flipping of training images.', action='store_true')
    # parser.add_argument('--keep_probability', type=float,
    #     help='Keep probability of dropout for the fully connected layer(s).', default=0.5)
    # parser.add_argument('--center_loss_factor', type=float,
    #     help='Center loss factor.', default=0.0)
    # learning rate and optimizer parameters.
    # parser.add_argument('--moving_average_decay', type=float,
    #     help='Exponential decay for tracking of training parameters.', default=0.9999)
    # parameters need no change.

    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    args = parse_arguments(sys.argv[1:])
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(parse_arguments(sys.argv[1:]))
