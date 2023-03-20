from util.dpt import Shot, ShotSet
from util.processors import *
import json
import os
import numpy as np

if __name__ == '__main__':
    east_data_dir = '../data/'
    east_raw = '/mypool/EAST/hdf5data'
    east_config_file = './config/local_config.json'
    with open(east_config_file) as json_file:
        config = json.load(json_file)

    shots, tags = config['shots'], config['tags']
    all_shots = shots['train'] + shots['val'] + shots['test']
    all_tags = tags['basic'] + tags['10k'] + tags['axuv']

    raw_set = ShotSet(os.path.join(east_data_dir, 'raw'), shot_list=all_shots)
    # raw_set = ShotSet(east_raw, shot_list=all_shots)
    dropped_set = raw_set.drop_signal(tags=all_tags, keep=True,
                                      output_directory=os.path.join(east_data_dir, 'dropped'))
    # resampled_basic = dropped_set.process(input_tags=tags['basic'] + tags['10k'] + tags['axuv'],
    #                                       processor=ResampleProcessor(1000),
    #                                       output_directory=os.path.join(east_data_dir, 'resampled_basic'))
    clipped_set = dropped_set.process(processor=ClipProcessor(start_time=2500E-3), input_tags=all_tags,
                                      output_directory=os.path.join(east_data_dir, 'clipped_first_false'))
    resampled_basic = clipped_set.process(input_tags=tags['basic'] + tags['10k'] + tags['axuv'],
                                          processor=ResampleProcessor(1000),
                                          output_directory=os.path.join(east_data_dir, 'resampled_basic'))


    def get_all_data(tags):
        all_data = dict()
        new_shot = Shot(shots['train'][0], clipped_set.input_directory)
        for tag in tags:
            all_data[tag] = new_shot.get(tag).raw
        for norm_shot in shots['train'][1:] + shots['val']:
            new_shot = Shot(norm_shot, clipped_set.input_directory)
            for tag in tags:
                all_data[tag] = np.concatenate((all_data[tag], new_shot.get(tag).raw))
        return all_data


    non_array_data = get_all_data(tags['basic'] + tags['10k'])
    axuv_data = get_all_data(tags['axuv'])

    axuv_tuple = tuple()
    for axuv_value in axuv_data.values():
        axuv_tuple += (axuv_value,)

    axuv_mean, axuv_std = float(np.concatenate(axuv_tuple).mean()), float(np.concatenate(axuv_tuple).std())
    norm_param = dict()

    for axuv_tag in tags['axuv']:
        norm_param[axuv_tag] = [axuv_mean, axuv_std]

    for non_array_key, non_array_item in non_array_data.items():
        mean, std = float(non_array_item.mean()), float(non_array_item.std())
        norm_param[non_array_key] = [mean, std]
    with open('./config/param.json', 'w') as f:
        json.dump(norm_param, f)

    normalized_set = resampled_basic.process(NormalizationProcessor(norm_param), input_tags=all_tags,
                                             output_directory=os.path.join(east_data_dir, 'normalized'))
    # stacked_axuv = stacked_sxr.process(StackProcessor(), input_tags=tags['axuv'], output_tag='axuv',
    #                                    output_directory=os.path.join(east_data_dir, 'stacked_axuv'))
    stacked_basic = normalized_set.process(StackProcessor(), input_tags=tags['basic'] + tags['10k'], output_tag='basic',
                                           output_directory=os.path.join(east_data_dir, 'stacked_basic'))
    stacked_axuv = stacked_basic.process(StackProcessor(), input_tags=tags['axuv'], output_tag='axuv',
                                         output_directory=os.path.join(east_data_dir, 'stacked_axuv')).drop_signal(
        all_tags)
    cut_train = stacked_axuv.process(CutProcessor(162, False), input_tags=['basic', 'axuv'],
                                     shot_filter=shots['train'],
                                     output_directory=os.path.join(east_data_dir, 'cut_train'))
    cut_val = stacked_axuv.process(CutProcessor(162, False), input_tags=['basic', 'axuv'],
                                   shot_filter=shots['val'],
                                   output_directory=os.path.join(east_data_dir, 'cut_val'))
    cut_test = stacked_axuv.process(CutProcessor(162, True), input_tags=['basic', 'axuv'],
                                    shot_filter=shots['test'],
                                    output_directory=os.path.join(east_data_dir, 'cut_test'))
    label_train = cut_train.process(BinaryLabelProcessor(), shot_filter=shots['train'], input_tags='basic',
                                    output_tag='label', output_directory=os.path.join(east_data_dir, 'label_train'))
    label_val = cut_val.process(BinaryLabelProcessor(), shot_filter=shots['val'], input_tags='basic',
                                output_tag='label', output_directory=os.path.join(east_data_dir, 'label_val'))
    label_test = cut_test.process(BinaryLabelProcessor(), shot_filter=shots['test'], input_tags='basic',
                                  output_tag='label', output_directory=os.path.join(east_data_dir, 'label_test'))
