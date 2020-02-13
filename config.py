class ModelBasic:
    couple_layers = 5
    in_out_dim = 28 * 28
    mid_dim = 1000
    hidden_dim = 5
    mask_config = 1


class DataBasic:
    dataset_dir = '/home/the/Datasets/MNIST/'


class TrainBasic:
    optimizer = 'adam'
    batch_size = 200
    dataset = 'biaobei'
    runname = ''
    checkpoint_dir = '/home/the/Projects/cache/nice/checkpoints'




    Momentum = 'momentum'
    Adam = 'adam'

