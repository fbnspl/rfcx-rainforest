import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import signal
tfd = tfp.distributions
import math as m

def freq_mask(input, param, name=None):
    """
    Apply masking to a spectrogram in the freq domain.
    Args:
      input: An audio spectogram.
      param: Parameter of freq masking.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """
    # TODO: Support audio with channel > 1.
    freq_max = tf.shape(input)[1]
    f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    f0 = tf.random.uniform(
        shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(freq_max), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, 0., input)


def time_mask(input, param, name=None):
    """
    Apply masking to a spectrogram in the time domain.
    Args:
      input: An audio spectogram.
      param: Parameter of time masking.
      name: A name for the operation (optional).
    Returns:
      A tensor of spectrogram.
    """
    # TODO: Support audio with channel > 1.
    time_max = tf.shape(input)[0]
    t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    t0 = tf.random.uniform(
        shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(time_max), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
    )
    return tf.where(condition, 0., input)


def mixup(x, l, beta):
    mix = tfd.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1])
    mix = tf.maximum(mix, 1 - mix)
    xmix = x * mix + x[::-1] * (1 - mix)
    #lmix = l * mix[:, :, 0, 0] + l[::-1] * (1 - mix[:, :, 0, 0])
    lmix = l * mix + l[::-1] * (1 - mix)
    return xmix, lmix


def mixup_one_hot(x, l, beta):
    # beta = tf.cast(beta, tf.float32)
    # x = tf.cast(x, tf.float32)
    # l = tf.cast(l, tf.float32)
    
    mix = tfd.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1])
    mix = tf.maximum(mix, 1 - mix)
    mix = tf.cast(mix, tf.float32)
    
    xmix = x * mix + x[::-1] * (1 - mix)
    lmix = l * mix[:, :, 0] + l[::-1] * (1 - mix[:, :, 0])
    #lmix = l * mix + l[::-1] * (1 - mix)
    return xmix, lmix

@tf.function
def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Examples
    --------
    >>> # Play at double speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
    >>> y_fast  = librosa.istft(D_fast, hop_length=512)

    >>> # Or play at 1/3 speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
    >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        STFT matrix

    rate :  float > 0 [scalar]
        Speed-up factor: `rate > 1` is faster, `rate < 1` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.

        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    Returns
    -------
    D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
        time-stretched STFT
    """

    n_fft = 2 * (D.shape[0] - 1)
    tf_pi = tf.constant(m.pi)

    if hop_length is None:
        hop_length = tf.constant(n_fft // 4)

    time_steps = tf.range(start=0, limit=D.shape[1], 
                          delta=rate, dtype=tf.float32)

    # Create an empty output array
    d_stretch = tf.zeros((D.shape[0], tf.size(time_steps)), D.dtype)

    # Expected phase advance in each bin
    phi_advance = tf.linspace(0.0, 
                              tf_pi * tf.cast(hop_length, tf.float32), 
                              D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = tf.math.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = tf.pad(D, [[0, 0], [0, 2]], mode='CONSTANT', constant_values=0)
    
    def accumulate_phase(step, D):
        step_idx = tf.cast(step, tf.int32)
        columns = D[:, step_idx:step_idx + 2]
        # Compute phase advance
        dphase = (tf.math.angle(columns[:, 1])
                  - tf.math.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * tf_pi * tf.math.round(dphase / (2.0 * tf_pi))

        # Accumulate phase
        return phi_advance + dphase
    
    # cumulate phase along time axis
    phase_acc_vec = tf.map_fn(fn=lambda t: accumulate_phase(t, D), 
                              elems=time_steps)
    phase_acc_cum = tf.math.cumsum(phase_acc_vec, axis=0)
    
    def stretch_d(step_phase, D):
        step, phase = step_phase
        step_idx = tf.cast(step, tf.int32)
        columns = D[:, step_idx:step_idx + 2]

        # Weighting for linear magnitude interpolation
        alpha = tf.math.mod(step, 1.0)
        mag = ((1.0 - alpha) * tf.math.abs(columns[:, 0])
               + alpha * tf.math.abs(columns[:, 1]))
        # stretch for time step
        return tf.cast(mag, tf.complex64)*tf.math.exp(tf.complex(0.,1.)*tf.cast(phase, tf.complex64))
    
    d_stretch = tf.map_fn(fn=lambda elem: stretch_d(elem, D), 
                          elems=(time_steps, phase_acc_cum), 
                          fn_output_signature=tf.complex64)
    return d_stretch

def time_stretch(y, frame_length, frame_step, rate):
    '''Time-stretch an audio series by a fixed rate.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    rate : float > 0 [scalar]
        Stretch factor.  If `rate > 1`, then the signal is sped up.

        If `rate < 1`, then the signal is slowed down.

    Returns
    -------
    y_stretch : np.ndarray [shape=(rate * n,)]
        audio time series stretched by the specified rate

    See Also
    --------
    pitch_shift : pitch shifting
    librosa.core.phase_vocoder : spectrogram phase vocoder


    Examples
    --------
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_fast = librosa.effects.time_stretch(y, 2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, 0.5)

    '''

    if rate <= 0:
        raise ParameterError('rate must be a positive number')
    y_padded = tf.pad(y, [[int(frame_length // 2), int(frame_length // 2)]], mode='REFLECT')
    stft = signal.stft(y_padded, frame_length, frame_step)
    stft = tf.transpose(stft)
    
    print(stft.shape)
    stft_stretch = phase_vocoder(stft, rate, frame_step)
    stft = tf.transpose(stft)
    print(stft_stretch.shape)
    y_stretch = signal.inverse_stft(stft_stretch, 
                                    frame_length, 
                                    frame_step,
                                    window_fn=signal.inverse_stft_window_fn(
                                        frame_step))
    print(y_stretch.shape)
    y_stretch = y_stretch[frame_length//2:]
    
    # fix length
    y_stretch = y_stretch[:int(1/rate*y.shape[0])]

    return y_stretch

def pitch_shift(y, sr, n_steps, bins_per_octave=12):
    '''Pitch-shift the waveform by `n_steps` half-steps.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time-series

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    n_steps : float [scalar]
        how many (fractional) half-steps to shift `y`

    bins_per_octave : float > 0 [scalar]
        how many steps per octave


    Returns
    -------
    y_shift : np.ndarray [shape=(n,)]
        The pitch-shifted audio time-series


    See Also
    --------
    time_stretch : time stretching
    phase_vocoder : spectrogram phase vocoder


    Examples
    --------
    Shift up by a major third (four half-steps)

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)

    Shift down by a tritone (six half-steps)

    >>> y_tritone = librosa.effects.pitch_shift(y, sr, n_steps=-6)

    Shift up by 3 quarter-tones

    >>> y_three_qt = librosa.effects.pitch_shift(y, sr, n_steps=3,
    ...                                          bins_per_octave=24)
    '''

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer):
        raise ParameterError('bins_per_octave must be a positive integer.')

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = core.resample(time_stretch(y, rate), float(sr) / rate, sr)

    # Crop to the same dimension as the input
    return util.fix_length(y_shift, len(y))




