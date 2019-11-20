Source Codes for Open-set structure recognition.
The files are organized as follows:

- data_gene.py/list_gene.py: generate the necessary files for training and testing, including class seperation and dataset split.
- sphere_network.py: model definition.
- train_double.py: training function of the open-set method.
- lfw.py: testing configuration, including the evaluation metric and data compilation.
- test.py: the function for testing the model performance using verification tasks.
- utils.py: utility function for training and testing. 
- main.py: the main running script for both the training and testing. 


##Train and Test
you can run the program by:
python main.py [multiple hyperparamters]


'--raw_data', type=str, help='Directory where raw data lies.', default='./extracted_data.npz'


'--num_class_train', type=int,help='Number of class for training.', default=17
'--num_class_test', type=int,
    help='Number of class for testing.', default=6


'--num_class', type=int,
    help='Number of class.', default=23


'--num_sample_per_class', type=int,
    help='Number of class.', default=50


'--test_list_dir', type=str, 
    help='test verification dir.', default='./test.txt'


'--train_data_dir', type=str,
    help='Path to the train data directory.',
    default='./train/'


'--test_data_dir', type=str,
    help='Path to the test data directory.',
    default='./test/'


'--train_list_dir', type=str,
    help='Image list file for training', default = './train.txt'


'--logs_base_dir', type=str, 
    help='Directory where to write event logs.', default='logs/demo_logs'


'--models_base_dir', type=str,
    help='Directory where to write trained models and checkpoints.', default='models/demo_model'


'--images_per_class', type=int,
    help='Number of images per class.', default=20


'--class_per_batch', type=int,
    help='Number of classes per batch.', default=5


'--pretrained_model', type=str,
    help='Load a pretrained model before training starts.'


'--epoch_size', type=int,
    help='Number of batches per epoch.', default=600


'--max_nrof_epochs', type=int,
    help='Number of epochs to run.', default=100


'--image_height', type=int,
    help='Image size (height, width) in pixels.', default=40


'--image_size', type=int,
    help='Image size.', default=40


'--image_width', type=int,
    help='Image size (height, width) in pixels.', default=40


'--center_loss_alfa', type=float,
    help='Center update rate for center loss.', default=0.05


'--learning_rate_decay_epochs', type=int,
    help='Number of epochs between learning rate decay.', default=20


'--learning_rate_decay_factor', type=float,
    help='Learning rate decay factor.', default=0.9)


'--optimizer', type=str, choices=['ADAGRAD', 'ADAM',  'MOM','SGD'],
    help='The optimization algorithm to use', default='ADAM'


'--center_weighting', type=float,
    help='weights for balancing the center loss and other loss.', default=1.0)


'--loss_type', type=str, choices=['softmax', 'lmcl', 'center', 'lmccl'], 
    help='Which type loss to be used.',default='lmccl'


'--network', type=str,
    help='which network is used to extract feature.',default='sphere_network'


'--gpu', type=str,
    help='which gpu to use.', default='2'


'--gpu_memory_fraction', type=float,
    help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9


'--alpha', type=float,
    help='Margin for cos margin.', default=0.35


'--scale', type=float,
    help='Scale as the fixed norm of weight and feature.', default=64.


'--embedding_size', type=int,
    help='Dimensionality of the embedding.', default=1024


'--fc_bn', 
    help='Wheater use bn after fc.', type = bool, default=True


'--weight_decay', type=float,
    help='L2 weight regularization.', default=0.0001


'--learning_rate', type=float,
    help='Initial learning rate.', default=0.001


'--seed', type=int,
    help='Random seed.', default=666


'--test_batch_size', type=int,
    help='Number of images to process in a batch in the test set.', default=200


'--test_nrof_folds', type=int,
    help='Number of folds to use for cross validation. Mainly used for testing.', default=10

    
'--model', type=str, default = '',
    help='Could be either a directory containing the meta_file and ckpt_file for testing.'