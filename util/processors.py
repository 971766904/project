from util.dpt import Signal, Processor
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d


class ResampleProcessor(Processor):
    def __init__(self, sample_rate):
        super().__init__()
        self.processor_kwargs.update({"SampleRate": sample_rate})

    def transform(self, signal: Signal):
        new_signal = deepcopy(signal)

        start_time = new_signal.prop_dict["StartTime"]
        down_time = start_time + len(new_signal.raw) / new_signal.prop_dict['SampleRate']
        down_time = round(down_time, 3)

        f = interp1d(np.linspace(start_time, down_time, len(new_signal.raw)), new_signal.raw)
        new_time = np.linspace(start_time, down_time,
                               int(round((down_time - start_time) * self.processor_kwargs["SampleRate"])))
        new_signal.raw = f(new_time)
        new_signal.prop_dict['SampleRate'] = len(new_signal.raw) / (down_time - start_time)
        return new_signal


class NormalizationProcessor(Processor):
    def __init__(self, param_dict):
        super().__init__()
        self.processor_kwargs.update({"ParamDict": param_dict})

    def transform(self, signal: Signal) -> Signal:
        new_signal = deepcopy(signal)
        mean = self.processor_kwargs["ParamDict"][signal.tag][0]
        std = self.processor_kwargs["ParamDict"][signal.tag][1]
        new_signal.raw = (new_signal.raw - mean) / std
        new_signal.raw = np.clip(new_signal.raw, -10, 10)

        return new_signal


class ClipProcessor(Processor):
    def __init__(self, start_time):
        super().__init__()
        self.processor_kwargs.update({"StartTime": start_time})

    def transform(self, signal: Signal) -> Signal:
        start_time = self.processor_kwargs['StartTime']
        end_time = self.processor_kwargs['t_end']
        end_time = round(end_time, 3)

        new_signal = deepcopy(signal)
        raw_time = new_signal.time
        new_signal.raw = new_signal.raw[(start_time <= raw_time) & (raw_time <= end_time)]
        new_signal.prop_dict['StartTime'] = start_time
        new_signal.prop_dict['SampleRate'] = len(new_signal.raw) / (end_time - start_time)
        return new_signal


class SliceProcessor(Processor):
    def __init__(self, window_length: int, overlap: float):
        super().__init__()
        assert (0 < overlap < 1), "Overlap is not between 0 and 1."
        self.processor_kwargs.update({"WindowLength": window_length,
                                      "Overlap": overlap})

    def transform(self, signal: Signal) -> Signal:
        window_length = self.processor_kwargs["WindowLength"]
        overlap = self.processor_kwargs["Overlap"]
        new_signal = deepcopy(signal)
        raw_sample_rate = new_signal.prop_dict["SampleRate"]
        step = round(window_length * (1 - overlap))

        down_time = new_signal.time[-1]

        down_time = round(down_time, 3)

        idx = len(signal.raw)
        window = list()
        while (idx - window_length) >= 0:
            window.append(new_signal.raw[idx - window_length:idx])
            idx -= step
        window.reverse()
        new_signal.prop_dict['SampleRate'] = raw_sample_rate * len(window) / (len(new_signal.raw) - window_length + 1)
        new_signal.raw = np.array(window)
        new_start_time = down_time - len(window) / new_signal.prop_dict['SampleRate']
        new_signal.prop_dict['StartTime'] = round(new_start_time, 3)
        return new_signal


class StackProcessor(Processor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        return Signal(np.row_stack([sig.raw for sig in signal.__iter__()]).T, signal.__getitem__(0).prop_dict)


class PadProcessor(Processor):
    def __init__(self, total_length: int):
        super().__init__()
        self.processor_kwargs.update({"TotalLength": total_length})

    def transform(self, signal: Signal) -> Signal:
        total_length = self.processor_kwargs["TotalLength"]
        new_signal = deepcopy(signal)
        new_signal.raw = np.pad(new_signal.raw, ((0, 0), (0, total_length - new_signal.raw.shape[-1])),
                                mode='constant', constant_values=(0, 0))
        return new_signal


class SliceBeforeLabelProcessor(Processor):
    def __init__(self, time_step, pre_time, is_test=False):
        super().__init__()
        self.processor_kwargs.update({"TimeStep": time_step,
                                      "PreTime": pre_time,
                                      "IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        time_step = self.processor_kwargs["TimeStep"]
        pre_time = self.processor_kwargs["PreTime"]
        is_test = self.processor_kwargs["IsTest"]
        is_disrupt = self.processor_kwargs["Is_disruptive"]
        new_signal = deepcopy(signal)

        if is_disrupt:
            if not is_test:
                new_signal.raw = new_signal.raw[-pre_time:]
        sliced_signal = SliceProcessor(time_step, (time_step - 1) / time_step).transform(new_signal)
        sliced_signal.prop_dict['PreTime'] = pre_time
        return sliced_signal


class CutProcessor(Processor):
    def __init__(self,  pre_time, is_test=False):
        super().__init__()
        self.processor_kwargs.update({"PreTime": pre_time,
                                      "IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        pre_time = self.processor_kwargs["PreTime"]
        is_test = self.processor_kwargs["IsTest"]
        is_disrupt = self.processor_kwargs["Is_disruptive"]
        new_signal = deepcopy(signal)

        if is_disrupt:
            if not is_test:
                new_signal.raw = new_signal.raw[-pre_time:]
        sliced_signal = new_signal
        sliced_signal.prop_dict['PreTime'] = pre_time
        return sliced_signal


class BinaryLabelProcessor(Processor):
    def __init__(self, is_test: bool = False):
        super().__init__()
        self.processor_kwargs.update({"IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        disruptive = self.processor_kwargs["Is_disruptive"]
        is_test = self.processor_kwargs["IsTest"]
        prop_dict = signal.prop_dict
        pre_time = signal.prop_dict['PreTime']
        if disruptive:
            if is_test:
                if len(signal.raw) < pre_time:
                    label = Signal(prop_dict=prop_dict, raw=np.ones(len(signal.raw)))
                else:
                    label = Signal(prop_dict=prop_dict,
                                   raw=np.concatenate((np.zeros(len(signal.raw) - pre_time), np.ones(pre_time))))
            else:
                label = Signal(prop_dict=prop_dict, raw=np.ones(len(signal.raw)))
        else:
            label = Signal(prop_dict=prop_dict, raw=np.zeros(len(signal.raw)))
        return label


class SoftLabelProcessor(Processor):
    def __init__(self, is_test=False):
        super().__init__()
        self.processor_kwargs.update({"IsTest": is_test})

    def transform(self, signal: Signal) -> Signal:
        pre_time = signal.prop_dict['PreTime']
        is_test = self.processor_kwargs["IsTest"]
        disruptive = self.processor_kwargs["Is_disruptive"]
        prop_dict = signal.prop_dict

        def sigmoid(x):
            sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
            return sig

        if disruptive:
            sigmoid_x = np.linspace(-10, 10, pre_time // 2)
            label_sigmoid = sigmoid(sigmoid_x)
            label_one = np.ones(pre_time - pre_time // 2)
            if is_test:
                label_zero = np.zeros(len(signal.raw) - pre_time)
                label = Signal(prop_dict=prop_dict, raw=np.concatenate((label_zero, label_sigmoid, label_one)))
            else:
                label = Signal(prop_dict=prop_dict, raw=np.concatenate((label_sigmoid, label_one)))
        else:
            label = Signal(prop_dict=prop_dict, raw=np.zeros(len(signal.raw)))
        return label


class ConvStackProcessor(Processor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        stacked_data = list()
        prop_dict = signal[0].prop_dict
        for sig in signal:
            stacked_data.append(sig.raw)
        stacked_data = np.array(stacked_data).transpose((1, 2, 0, 3))
        return Signal(raw=stacked_data, prop_dict=prop_dict)
