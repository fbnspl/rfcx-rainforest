import tensorflow as tf
import numpy as np


"""
Hello everyone! Thank you for your interest. I just wanted to make a clarification about train_tp.csv vs. train_fp.csv.
Training data for the competition was collected using a rudimentary detection algorithm, where an example of each species/songtype was cross-correlated with a large amount of soundscape audio, and potential matches were flagged and reviewed by experts.
Each row in train_tp.csv corresponds to a flagged segment of one of the training audio files, that was verified by experts to contain the species_id and songtype_id.
Each row in train_fp.csv corresponds to a flagged segment that was verified to not contain the species and songtype.

0 species_id - unique identifier for species
1 songtype_id - unique identifier for songtype
2 t_min - start second of annotated signal
3 f_min - lower frequency of annotated signal
4 t_max - end second of annotated signal
5 f_max- upper frequency of annotated signal
6 is_tp- [tfrecords only] an indicator of whether the label is from the train_tp (1) or train_fp (0) file.
# e.g. 14, 1, 44.544, 2531.25, 45.1307, 5531.25, 1
"""

def get_species_times(label_info):
    """
    Get the species onset and offset timings from the label_info string
    """
    # decode to string
    label_info = label_info.numpy().decode()
    # remove front and trailing characters
    label_info = label_info.replace('"', '').replace('\n', '')
    label_blocks = label_info.split(';')
    species_id, onset, offset = [], [], []
    for label in label_blocks:
        label = label.split(',')
        # parameter 6 denotes whether the label is from the train_tp (1) or train_fp (0) file
        if int(label[6]) == 1:
            species_id.append(int(label[0]))
            onset.append(float(label[2]))
            offset.append(float(label[4]))
    return list(zip(species_id, onset, offset))


def get_binary_y(label_info, config):
    dt = config['duration'] / config['n_frames']
    
    def get_y_from_label_info(label_info):
        time_pattern = get_species_times(label_info)
        y_frames = np.zeros((config['n_frames'], config['num_species']), dtype=np.uint8)

        for species_id, onset, offset in time_pattern:

            event_start = int(np.round(onset / dt))
            event_end = int(np.round(offset / dt))
            if event_start < config['n_frames']:
                if event_end >= config['n_frames']:
                    event_end = config['n_frames'] - 1
                y_frames[event_start:event_end + 1, species_id] = 1

            else:
                pass

        return y_frames
    y_frames = tf.py_function(get_y_from_label_info, [label_info], tf.uint8)
    y_frames = tf.convert_to_tensor(y_frames)
    y_frames.set_shape((config['n_frames'], config['num_species']))
    return y_frames

def get_one_hot_y(label_info, config):    
    def get_y_from_label_info(label_info):
        time_pattern = get_species_times(label_info)
        y_frames = np.zeros((config['num_species']), dtype=np.uint8)
        for species_id, _, _ in time_pattern:
                y_frames[species_id] = 1
        return y_frames
    y_frames = tf.py_function(get_y_from_label_info, [label_info], tf.uint8)
    y_frames = tf.convert_to_tensor(y_frames)
    y_frames.set_shape((config['num_species']))
    return y_frames


def read_train_tfrec(example):
    tfrec_format = {
        'recording_id': tf.io.FixedLenFeature([], tf.string),
        'label_info': tf.io.FixedLenFeature([], tf.string),
        'audio_wav': tf.io.FixedLenFeature([], tf.string)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    
    example["audio_wav"], example["sample_rate"] = tf.audio.decode_wav(example["audio_wav"])

    return example["audio_wav"], example["label_info"], tf.cast(example["sample_rate"], tf.int64), example["recording_id"]

def read_test_tfrec(example):
    tfrec_format = {
        'recording_id': tf.io.FixedLenFeature([], tf.string),
        'audio_wav': tf.io.FixedLenFeature([], tf.string)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    
    example["audio_wav"], example["sample_rate"] = tf.audio.decode_wav(example["audio_wav"])

    return example["audio_wav"], tf.cast(example["sample_rate"], tf.int64), example["recording_id"]