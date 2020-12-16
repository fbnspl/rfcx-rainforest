import tensorflow as tf
import numpy as np
import librosa



def get_config():
    # get all unique ids
    spec_ids = [f for f in range(24)] 

    # best-practice: write down your preprocessing config in a dictonary
    config = {'sr': 48000, 
              'audio_length': 1,
              'mono': True,
              'n_mels': 64,
              'n_fft': 2000,
              'hop_length': 500,
              'win_length': 2000,
              'window': 'hann',
              'center': True,
              'pad_mode': 'reflect',
              'power': 2.0,
              'classes': spec_ids
             }

    # save number of frames from length in samples divided by fft hop length
    config['n_frames'] = int(config['sr']*config['audio_length']/config['hop_length']) + 1

    # save input shape for model
    config['input_shape'] = (config['n_mels'], config['n_frames'], 1)
    
    return config


def save_config(config, fp):
    # save config 
    with open('data/yamnet_embeddings_active/yamnet_config.json', 'w+') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)

    # pretty print json
    print(json.dumps(config, indent=4))


def folder_name_to_one_hot(file_path):
    
    # for example: _data/TinyUrbanSound8k/train/siren/157648-8-0-0_00.wav
    label = Path(file_path).parts[-2]
    label_idx = classes.index(label)
    
    # get one hot encoded array
    one_hot = tf.one_hot(label_idx, len(config['classes']), on_value=None, off_value=None, 
                         axis=None, dtype=tf.uint8, name=None)
    return one_hot


def audiopath_to_melspec(file_path):
    # load audio data 
    y, _ = librosa.core.load(file_path, sr=config['sr'], mono=config['mono'], offset=0.0, duration=None, 
                             dtype=np.float32, res_type='kaiser_best')

    # calculate stft from audio data
    stft = librosa.core.stft(y, n_fft=config['n_fft'], hop_length=config['hop_length'], 
                             win_length=config['win_length'], window=config['window'], 
                             center=config['center'], dtype=np.complex64, pad_mode=config['pad_mode'])

    # generate mel-filter matrix
    mel_filter = librosa.filters.mel(config['sr'], 
                                     config['n_fft'], 
                                     n_mels=config['n_mels'], 
                                     fmin=0.0, 
                                     fmax=None, 
                                     htk=False, 
                                     norm='slaney', 
                                     dtype=np.float32)
    
    # filter stft with mel-filter
    mel_spec = mel_filter.dot(np.abs(stft).astype(np.float32) ** config['power'])
    
    # add channel dimension for conv layer  compatibility
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    
    mel_spec_frames = librosa.util.frame(mel_spec, frame_length=2048, hop_length=64)
    
    return mel_spec_frames


def load_and_preprocess_data(file_path):
    # path string is saved as byte array in tf.data.dataset -> convert back to str
    if type(file_path) is not str:
        file_path = file_path.numpy()
        file_path = file_path.decode('utf-8')
    
    # get malspec
    mel_spec = audiopath_to_melspec(file_path)
    
    # get ground truth from file_path string
    one_hot = folder_name_to_one_hot(file_path)
    
    return mel_spec, one_hot


# there is a TF bug where we get an error if the size of the tensor from a py.function is not set manualy
# when called from a map()-function.
def preprocessing_wrapper(file_path):
    mel_spec, one_hot = tf.py_function(load_and_preprocess_data, [file_path], [tf.float32, tf.uint8])
    
    mel_spec.set_shape([config['n_mels'], config['n_frames'], 1])
    one_hot.set_shape([len(config['classes'])])
    return mel_spec, one_hot