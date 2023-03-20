import tensorflow as tf
import tensorflow_io as tfio


def make_generator_from_directory(file_path: str):
    """Read .hdf5 file from given path and return a generator containing inputs and outputs.

    Args:
        file_path (str): The path of the file to be read
    Yields:
        tuple: (inputs, output)
    """
    basic = tfio.IODataset.from_hdf5(file_path, '/data/basic', spec=tf.float32, name='basic')
    conv = tfio.IODataset.from_hdf5(file_path, '/data/axuv', spec=tf.float32, name='axuv')
    label = tfio.IODataset.from_hdf5(file_path, '/data/label', spec=tf.float32, name='label')
    basic_gen = basic.as_numpy_iterator()
    conv_gen = conv.as_numpy_iterator()
    label_gen = label.as_numpy_iterator()
    for i1, i2, l in zip(conv_gen, basic_gen, label_gen):
        yield {"input_1": i1, "input_2": i2}, l


def make_dataset_from_generator(file_path: str):
    """Make tf.data.Dataset from generator, corresponding to the given path.

    Args:
        file_path (str): The path of the file to be read

    Returns:
        tf.data.Dataset: dataset instance according to the generator
    """
    dataset = tf.data.Dataset.from_generator(lambda: make_generator_from_directory(file_path),
                                             output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.int32),
                                             output_shapes=({"input_1": tf.TensorShape([16]),
                                                             "input_2": tf.TensorShape([9])}, ()))
    return dataset


def concat_dataset_from_directory(directory: str):
    """Concatenate datasets built from given directory.

    Args:
        directory (str): Directory given to build the dataset, contains multiple files

    Returns:
         tf.data.Dataset: concatenated datasets
    """
    list_dataset = tfio.IODataset.list_files(str(directory + '/*/*.hdf5'))
    file_path_0 = list_dataset.take(1).as_numpy_iterator().next()
    dataset = make_dataset_from_generator(file_path_0)
    for sub in list_dataset.take(-1):
        file_path = sub.numpy()
        if file_path == file_path_0:
            continue
        else:
            dataset = dataset.concatenate(make_dataset_from_generator(file_path))

    return dataset


def cal_ds_length(ds: tf.data.Dataset):
    """Calculate the length (or number of samples) of the dataset given.

    Returns:
        int: length of the dataset
    """
    return ds.reduce(0, lambda x, _: x + 1).numpy()


def make_ds(directory: str):
    """

    Args:
        directory (str): Directory given to build the dataset, contains multiple files

    Returns:
        tf.data.Dataset: concatenated and shuffled dataset
    """
    ds = concat_dataset_from_directory(directory)
    # ds_length = cal_ds_length(ds)
    # ds = ds.shuffle(ds_length)
    return ds
