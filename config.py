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
    dataset = 'biaobei'
    latent = 'logistic'
    batch_size = 200
    runname = ''
    checkpoint_dir = '/home/the/Projects/cache/nice/checkpoints'
    filename = '%s_' % dataset \
               + 'bs%d_' % batch_size \
               + '%s_' % latent \
               + 'cp%d_' % ModelBasic.couple_layers \
               + 'md%d_' % ModelBasic.mid_dim \
               + 'hd%d_' % ModelBasic.hidden_dim



    Momentum = 'momentum'
    Adam = 'adam'

