from typing import Optional
import warnings
import numpy as np
import os
import h5py
import math
from copy import deepcopy


class Signal(object):
    """Signal object.

    Stores data and necessary parameters of a signal.

    Attributes:
        raw (numpy.ndarray): Raw data of the Signal instance.
        prop_dict (dict): Necessary attributes of the Signal instance. Note that SampleRate and StartTime will be set default if not given.
        tag (str): Name of the signal. Optional.
    """

    def __init__(self, raw: np.ndarray, prop_dict, tag: Optional[str] = None):
        self.raw = raw
        self.prop_dict = prop_dict
        self.tag = tag

        try:
            self.prop_dict['SampleRate']
        except TypeError:
            self.prop_dict['SampleRate'] = 1
            warnings.warn("SampleRate not in props. Set to 1.")
        try:
            self.prop_dict['StartTime']
        except KeyError:
            self.prop_dict['StartTime'] = 0
            warnings.warn("StartTime not in props. Set to 0.")

    @property
    def time(self):
        """
        Returns:
            numpy.ndarray: Time axis of the Signal instance according to SampleRate and StartTime given.
        """
        start_time = self.prop_dict['StartTime']
        sample_rate = self.prop_dict['SampleRate']
        down_time = start_time + len(self.raw) / sample_rate
        return np.linspace(start_time, down_time, len(self.raw))


class Processor(object):
    """Process signals.

    Process signals according to the ``transform()`` method.

    """

    def __init__(self, **processor_kwargs):
        self.processor_kwargs = processor_kwargs

    def transform(self, *signal: Signal) -> Signal:
        """Process signal.

        This is used for subclassed processors.
        To use it, override the method.

        Args:
            signal: signal(s) to be processed.

        Returns:
            New signal processed.
        """
        pass


class Shot(object):
    """Shot object.

    Stores multiple Signal instances as data, and other necessary parameters of a shot.

    Attributes:
        shot (Any): Number of the shot.
        directory (str): Directory where the shot(s) is(are) saved.
    """

    def __init__(self, shot, directory: Optional[str] = None):
        self.shot = int(shot)
        self.__directory = directory
        self.__new_signal = dict()
        if self.__directory is not None and os.path.exists(self.__directory):
            path = os.path.join(self.__directory, str(math.floor(self.shot / 100) * 100), '{}.hdf5'.format(self.shot))
            try:
                hdf5_file = h5py.File(path, 'r+')
                self.__file = hdf5_file
            except OSError as e:
                print(e)
                self.__file = None
        else:
            self.__file = None
        if self.__file:
            meta_dict = dict()
            self.__origin_signal = list(self.__file.get('data').keys())
            for meta_tag in self.__file.get('meta').keys():
                try:
                    meta_dict[meta_tag] = deepcopy(self.__file.get('meta').get(meta_tag)[:][0])
                except ValueError:
                    meta_dict[meta_tag] = deepcopy(self.__file.get('meta').get(meta_tag)[()])
            self.meta = meta_dict
        else:
            self.__origin_signal = list()
            self.meta = dict()

    @property
    def data(self):
        """
        Returns:
            set: All tags of the signals stored within the shot.
        """
        return set(self.__origin_signal + list(self.__new_signal.keys()))

    def add(self, tag: str, signal: Signal):
        """Add a new signal to the shot.

        The method does not change any file in disk.

        Args:
            tag (str): Name of the signal to be added.
            signal (Signal): Signal instance to be added.
        Raises:
            ValueError: if the signal to be added is already in data.
            TypeError: if the signal is not an instance of Signal.
        """

        if tag in (self.__origin_signal + list(self.__new_signal.keys())):
            raise ValueError("{} is already in data.".format(tag))
        else:
            if isinstance(signal, Signal):
                self.__new_signal[tag] = signal
            else:
                raise TypeError('\'signal\' is not an instance of Signal')

    def drop(self, tags, keep=False):
        """Remove (or keep) existing signal(s) from the shot.

        The method does not change any file in disk.

        Args:
            tags: Name of the signal(s) to be removed.
            keep (bool): Whether to keep the tags or not. Default False.
        Raises:
            ValueError: if any of the signal(s) to be removed (or kept) is not found in data.
        """
        if isinstance(tags, str):
            tags = [tags]
        tags = set([tag for tag in tags])
        for tag in tags:
            if tag not in self.data:
                raise ValueError("{} is not found in {} data.".format(tag, self.shot))
        if keep:
            drop_tags = self.data.difference(tags)
        else:
            drop_tags = tags
        for tag in drop_tags:
            if tag in self.__origin_signal:
                self.__origin_signal.remove(tag)
            if tag in self.__new_signal.keys():
                del self.__new_signal[tag]

    def update(self, tag: str, signal: Signal):
        """Update an existing signal to the shot.

        The method does not change any file in disk.

        Args:
            tag (str): Name of the signal to be updated.
            signal (Signal): Signal instance to be updated.
        Raises:
            ValueError: if the signal to be updated is not found in data.
            TypeError: if the signal is not an instance of the signal.
        """
        if tag not in (self.__origin_signal + list(self.__new_signal.keys())):
            raise ValueError("{} is not found in data.".format(tag))
        else:
            if isinstance(signal, Signal):
                self.__new_signal[tag] = signal
            else:
                raise TypeError('\'signal\' is not an instance of Signal')

    def get(self, tag: str) -> Signal:
        """Get an existing signal of the shot.

        The method does not change any file in disk.
        First check through if the signal to be got is newly added or updated.
        If so, return the signal from RAM, else read the signal instance from disk to RAM.
        Note that data of the Signal instance stored in disk is not read to RAM until it is got.

        Args:
            tag (str): Name of the signal to be got.
        Returns:
            Signal: the signal instance to be got.
        Raises:
            ValueError: if the signal to be got is not found in data.
        """
        if tag in self.__new_signal.keys():
            return self.__new_signal[tag]
        elif tag in self.__origin_signal:
            prop_dict = dict()
            data = self.__file.get('data')
            dataset = data.get(tag)
            for prop_key, prop_value in dataset.attrs.items():
                prop_dict[prop_key] = deepcopy(prop_value)
            return Signal(prop_dict=prop_dict, raw=deepcopy(dataset[()]), tag=tag)
        else:
            raise ValueError("{} is not found in data.".format(tag))

    def process(self, processor: Processor, input_tags, output_tag: Optional[str] = None):
        if isinstance(input_tags, str):
            input_tags = [input_tags]
        if not isinstance(input_tags, list):
            raise TypeError("Expected input_tags to be list or str, got {} instead".format(type(input_tags)))
        processor.processor_kwargs.update(self.meta)
        processor.processor_kwargs.update({"Shot": self.shot})

        if output_tag is None:
            for tag in input_tags:
                new_signal = processor.transform(self.get(tag))
                self.update(tag, new_signal)
        else:
            input_sigs = ()
            for input_tag in input_tags:
                input_sigs += (self.get(input_tag),)
            new_signal = processor.transform(*input_sigs)
            self.add(output_tag, new_signal)

    def save(self, new_directory: Optional[str] = None):
        """Save the shot to specified directory.

        Save all changes done before to the disk.
        Save the shot to specified directory. Please check carefully the new directory to save.
        If the specified directory is None or same to the original directory, changes will cover the original files.

        Args:
            new_directory: Directory specified to save the shot to disk. Default None.
        """
        tags = self.data

        if new_directory is not None and (new_directory != self.__directory):
            path = os.path.join(new_directory, str(math.floor(self.shot / 100) * 100))
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, '{}.hdf5'.format(self.shot))
            try:
                file = h5py.File(file_path, 'x')
            except OSError:
                os.remove(file_path)
                file = h5py.File(file_path, 'x')
            if self.__file is None:
                self.__file = file
            data = file.create_group('data')
            for data_tag in tags:
                signal = self.get(data_tag)
                dataset = data.create_dataset(data_tag, data=signal.raw)
                for key, value in signal.prop_dict.items():
                    dataset.attrs.create(key, value)
            meta = file.create_group('meta')
            new_meta_tag = ['IpMean', 'IsDisrupt', 'DownTime', 'StartTime']
            kt = 0
            for meta_tag in self.meta.keys():
                meta.create_dataset(meta_tag, data=self.meta[meta_tag])
                kt += 1
            file.close()
        else:
            data = self.__file.get('data')
            for data_tag in data.__iter__():
                if data_tag not in tags:
                    data.__delitem__(data_tag)
            for data_tag in self.__new_signal.keys():
                if data_tag in data.__iter__():
                    data.__delitem__(data_tag)
                signal = self.get(data_tag)
                dataset = data.create_dataset(data_tag, data=signal.raw)
                for key, value in signal.prop_dict.items():
                    dataset.attrs.create(key, value)

        self.__file.close()


class ShotSet(object):
    def __init__(self, input_directory: str, shot_list: Optional[list] = None):
        """ShotSet object.

        Stores multiple shot files in the given directory. ShotSet have to be saved to disk.

        Args:
            input_directory (str): root directory, with per 100 shots saved in subdirectories.
            shot_list (list): shot files to be processed, if None, process all shots under input_directory.
        """
        self.__input_directory = input_directory
        self.__shot_list = shot_list
        if self.__shot_list is None:
            self.__shot_list = list()
            for _, __, files in os.walk(self.__input_directory):
                for file in files:
                    if file.endswith(".hdf5"):
                        self.__shot_list.append(int(file.split('.')[0]))

    @property
    def input_directory(self):
        """
        Returns:
            str: input_directory of the ShotSet.
        """
        return self.__input_directory

    @property
    def shot_list(self):
        """
        Returns:
            list: shot files to be processed in the ShotSet.
        """
        return self.__shot_list

    def drop_shot(self, shot_list: list, keep=False):
        """Remove (or keep) shot files in the shot list.

        Args:
            shot_list (list): shot files to remove (or keep).
            keep (bool): whether to keep the files or to remove. Default False.
        """
        if keep:
            drop_list = set(self.shot_list).difference(set(shot_list))
            self.__shot_list = shot_list
        else:
            drop_list = set(shot_list)
            self.__shot_list = list(set(self.shot_list).difference(drop_list))

        for drop_shot in drop_list:
            shot_path = os.path.join(self.__input_directory, str(math.floor(drop_shot / 100) * 100),
                                     '{}.hdf5'.format(drop_shot))
            os.remove(shot_path)

    def drop_signal(self, tags, shot_filter: Optional[list] = None, keep: Optional[bool] = False,
                    output_directory: Optional[str] = None):
        if shot_filter is None:
            shot_filter = self.__shot_list
        for each_shot in shot_filter:
            shot = Shot(each_shot, self.__input_directory)
            shot.drop(tags=tags, keep=keep)

            if output_directory is not None and (output_directory != self.input_directory):
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            else:
                output_directory = self.input_directory
            shot.save(output_directory)

        if output_directory is None and shot_filter is None:
            return self
        else:
            return ShotSet(output_directory, shot_filter)

    def process(self, processor: Processor, input_tags: Optional = None, output_tag: Optional[str] = None,
                shot_filter: Optional[list] = None, output_directory: Optional[str] = None, **kwargs):
        processor.processor_kwargs.update(kwargs)
        if shot_filter is None:
            shot_filter = self.__shot_list
        for each_shot in shot_filter:
            shot = Shot(each_shot, self.__input_directory)
            if input_tags is None:
                input_tags = list(shot.data)
            try:
                shot.process(processor=processor, input_tags=input_tags, output_tag=output_tag)
            except Exception as e:
                print("Shot {}".format(each_shot))
                raise e

            if output_directory is not None and (output_directory != self.input_directory):
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
            else:
                output_directory = self.input_directory
            shot.save(output_directory)

        if output_directory is None and shot_filter is None:
            return self
        else:
            return ShotSet(output_directory, shot_filter)
