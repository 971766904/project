import tensorflow as tf
import tensorflow_io as tfio
import os


def make_generator_from_directory(file_path):
    """
    Read .hdf5 file from given path and return a generator containing inputs and outputs.

    Parameters
    ----------
    file_path: The path of the file to be read.

    Returns generator of (inputs, outputs) tuple.
    -------

    """
    basic = tfio.IODataset.from_hdf5(file_path, '/data/basic', spec=tf.float32, name='basic')
    axuv = tfio.IODataset.from_hdf5(file_path, '/data/axuv', spec=tf.float32, name='axuv')
    basic_gen = basic.as_numpy_iterator()
    axuv_gen = axuv.as_numpy_iterator()
    for i1, i2 in zip(axuv_gen, basic_gen):
        yield {"input_1": i1, "input_2": i2}, int(tf.strings.split(file_path, os.sep)[-3])


def make_dataset_from_generator(file_path):
    """
    Make tf.data.Dataset from generator, corresponding to the given path

    Parameters
    ----------
    file_path: The path of the file to be read

    Returns tf.data.Dataset
    -------

    """
    dataset = tf.data.Dataset.from_generator(lambda: make_generator_from_directory(file_path),
                                             output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.int32),
                                             output_shapes=({"input_1": tf.TensorShape([16, 1]),
                                                             "input_2": tf.TensorShape([9, 1])}, ()))
    return dataset


def concat_dataset_from_directory(directory):
    """
    Concatenate datasets built from given directory.

    Parameters
    ----------
    directory: Directory given to build the dataset, contains multiple files.

    Returns concatenated tf.data.Dataset.
    -------

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


def process_path(file_path):
    basic = tfio.IODataset.from_hdf5(file_path, '/data/basic', spec=tf.float32, name='basic')
    axuv = tfio.IODataset.from_hdf5(file_path, '/data/axuv', spec=tf.float32, name='axuv')
    label = tfio.IODataset.from_hdf5(file_path, '/data/label', spec=tf.int32, name='label')

    return {"input_1": axuv, "input_2": basic}, label

