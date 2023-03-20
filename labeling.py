from util.processors import SoftLabelProcessor, SliceBeforeLabelProcessor, ConvStackProcessor, BinaryLabelProcessor
from dataset import make_ds
from util.dpt import ShotSet, Shot
import os
import json
import tensorflow as tf

if __name__ == '__main__':
    jtext_data_dir = '/mypool/xfm/transfer_east/jtext_data'
    jtext_config_file = '/home/xfm/transfer_east_new/config.json'
    with open(jtext_config_file) as json_file:
        config = json.load(json_file)

    RNN_TIME_STEP = 32
   #  PRE_TIME = 2 * (20 + RNN_TIME_STEP)  # soft label for regression
    PRE_TIME = 20 + RNN_TIME_STEP  # binary label for classification

    shots = config['shots']
    tags = config['tags']
    input_tags = tags['10k'] + tags['50k'] + ['basic', 'sxr', 'axuv']
    conv_tags = tags['10k'] + tags['50k'] + ['sxr', 'axuv']

    padded_set_train = ShotSet(input_directory=os.path.join(jtext_data_dir, 'padded'), shot_list=shots['train'])
    padded_set_val = ShotSet(input_directory=os.path.join(jtext_data_dir, 'padded'), shot_list=shots['val'])
    padded_set_test = ShotSet(input_directory=os.path.join(jtext_data_dir, 'padded'), shot_list=shots['test'])


    def sep_disrupt(shot_set: ShotSet):
        disrupt_list = list()
        normal_list = list()
        for shot in shot_set.shot_list:
            if Shot(shot, shot_set.input_directory).meta["IsDisrupt"]:
                disrupt_list.append(shot)
            else:
                normal_list.append(shot)
        return disrupt_list, normal_list


    train_dis_list, train_non_list = sep_disrupt(padded_set_train)
    val_dis_list, val_non_list = sep_disrupt(padded_set_val)
    test_dis_list, test_non_list = sep_disrupt(padded_set_test)

    slice_before_label_set_train_dis = padded_set_train.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=False), input_tags=input_tags,
        shot_filter=train_dis_list, output_directory=os.path.join(jtext_data_dir, 'sliced_train', 'disruptive'))
    slice_before_label_set_train_non = padded_set_train.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=False), input_tags=input_tags,
        shot_filter=train_non_list, output_directory=os.path.join(jtext_data_dir, 'sliced_train', 'non_disruptive'))
    slice_before_label_set_val_dis = padded_set_val.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=False), input_tags=input_tags,
        shot_filter=val_dis_list, output_directory=os.path.join(jtext_data_dir, 'sliced_val', 'disruptive'))
    slice_before_label_set_val_non = padded_set_val.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=False), input_tags=input_tags,
        shot_filter=val_non_list, output_directory=os.path.join(jtext_data_dir, 'sliced_val', 'non_disruptive'))
    slice_before_label_set_test_dis = padded_set_test.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=True), input_tags=input_tags,
        shot_filter=test_dis_list, output_directory=os.path.join(jtext_data_dir, 'sliced_test', 'disruptive'))
    slice_before_label_set_test_non = padded_set_test.process(
        SliceBeforeLabelProcessor(time_step=RNN_TIME_STEP, pre_time=PRE_TIME, is_test=True), input_tags=input_tags,
        shot_filter=test_non_list, output_directory=os.path.join(jtext_data_dir, 'sliced_test', 'non_disruptive'))

    label_train_dis = slice_before_label_set_train_dis.process(
        BinaryLabelProcessor(is_test=False),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_train', 'disruptive'))

    label_train_non = slice_before_label_set_train_non.process(
        BinaryLabelProcessor(is_test=False),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_train', 'non_disruptive'))

    label_val_dis = slice_before_label_set_val_dis.process(
        BinaryLabelProcessor(is_test=False),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_val', 'disruptive'))

    label_val_non = slice_before_label_set_val_non.process(
        BinaryLabelProcessor(is_test=False),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_val', 'non_disruptive'))

    label_test_dis = slice_before_label_set_test_dis.process(
        BinaryLabelProcessor(is_test=True),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_test', 'disruptive'))

    label_test_non = slice_before_label_set_test_non.process(
        BinaryLabelProcessor(is_test=True),
        input_tags='basic', output_tag='label',
        output_directory=os.path.join(jtext_data_dir, 'label_test', 'non_disruptive'))

    final_train_dis = label_train_dis.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_train', 'disruptive')).drop_signal(conv_tags)
    final_train_non = label_train_non.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_train', 'non_disruptive')).drop_signal(conv_tags)
    final_val_dis = label_val_dis.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_val', 'disruptive')).drop_signal(conv_tags)
    final_val_non = label_val_non.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_val', 'non_disruptive')).drop_signal(conv_tags)
    final_test_dis = label_test_dis.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_test', 'disruptive')).drop_signal(conv_tags)
    final_test_non = label_test_non.process(
        ConvStackProcessor(), input_tags=conv_tags, output_tag='conv',
        output_directory=os.path.join(jtext_data_dir, 'final_test', 'non_disruptive')).drop_signal(conv_tags)

    tf.data.experimental.save(make_ds(os.path.join(jtext_data_dir, 'final_train', 'disruptive')),
                              os.path.join(jtext_data_dir, 'dataset', 'train_dis'))
    tf.data.experimental.save(make_ds(os.path.join(jtext_data_dir, 'final_train', 'non_disruptive')),
                              os.path.join(jtext_data_dir, 'dataset', 'train_non'))
    tf.data.experimental.save(make_ds(os.path.join(jtext_data_dir, 'final_val', 'disruptive')),
                              os.path.join(jtext_data_dir, 'dataset', 'val_dis'))
    tf.data.experimental.save(make_ds(os.path.join(jtext_data_dir, 'final_val', 'non_disruptive')),
                              os.path.join(jtext_data_dir, 'dataset', 'val_non'))
