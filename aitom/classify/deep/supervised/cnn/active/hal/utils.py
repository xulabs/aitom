import os


def create_save_folder(args, method):
    # create the checkpoint path:
    if not os.path.isdir(os.path.join(args.experiment_folder, 'models')):
        os.mkdir(os.path.join(args.experiment_folder, 'models'))
    model_folder = os.path.join(args.experiment_folder, 'models')

    checkpoint_path = os.path.join(model_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.hdf5'.format(
        alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
    ))
    # create the results path:
    if not os.path.isdir(os.path.join(args.experiment_folder, 'results')):
        os.mkdir(os.path.join(args.experiment_folder, 'results'))
    results_folder = os.path.join(args.experiment_folder, 'results')
    results_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.pkl'.format(
        alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
    ))
    # create the label entropy path:
    entropy_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}_entropy.pkl'.format(
        alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
    ))

    # create the results_path_each_iteration path:
    if not os.path.isdir(os.path.join(args.experiment_folder, 'labeled_sample_each_iteration')):
        os.mkdir(os.path.join(args.experiment_folder,
                              'labeled_sample_each_iteration'))
    results_path_each_iteration_folder = os.path.join(
        args.experiment_folder, 'labeled_sample_each_iteration')
    results_path_each_iteration = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}_labeled_index_each_iteration.pkl'.format(
        alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
    ))

    return checkpoint_path
