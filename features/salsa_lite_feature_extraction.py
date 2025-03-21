"""
This module extract salsa-lite and salsa-ipd feature for microphone array format.
"""
import os
import shutil

import fire
import h5py
import librosa
import numpy as np
import yaml
from timeit import default_timer as timer
from tqdm import tqdm

from salsa_feature_extraction import compute_scaler


def extract_features(data_dir: str = "../DCASE2024_SELD_dataset", feat_dir: str = "../DCASE2024_SELD_dataset/feat_labels",
                     audio_format: str = "mic", fs: int = 24000, n_fft: int = 512, hop_length: int = 300, win_length: int = 512,
                     fmin_doa: int = 50, fmax_doa: int = 2000,
                     feature_type: str = 'salsa_lite',
                     task: str = 'feature_scaler') -> None:
    """
    Extract salsa_lite and salsa_ipd features:
        salsa_lite: log-linear spectrogram + normalized interchannel phase difference
        salsa_ipd:  log-linear spectrogram + interchannel phase difference
    The frequency range of log-linear spectrogram is 0 to 9kHz.
    :param data_config: Path to data config file.
    :param feature_type: Choices are 'salsa_lite', 'salsa_ipd'
    :param task: 'feature_scaler': extract feature and scaler, 'feature': only extract feature, 'scaler': only extract
        scaler.
    """


    # Doa info
    n_mics = 4
    fmax_doa = np.min((fmax_doa, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = int(np.floor(fmin_doa * n_fft / np.float(fs)))  # 512: 1; 256: 0
    upper_bin = int(np.floor(fmax_doa * n_fft / np.float(fs)))  # 9000Hz: 512: 192, 256: 96
    lower_bin = np.max((1, lower_bin))

    # Cutoff frequency for spectrograms
    fmax = 9000  # Hz
    cutoff_bin = int(np.floor(fmax * n_fft / np.float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
    assert upper_bin <= cutoff_bin, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

    # Normalization factor for salsa_lite
    c = 343
    delta = 2 * np.pi * fs / (n_fft * c)
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1

    # Get feature descriptions
    feature_description = '{}fs_{}nfft_{}nhop_{}fmaxdoa'.format(fs, n_fft, hop_length, int(fmax_doa))
    print('Feature description: {}'.format(feature_description))

    assert audio_format == 'mic', 'SALSA-Lite and SALSA-IPD are only for MIC format!'
    splits = ['mic_dev']

    # Extract features
    if task in ['feature_scaler', 'feature']:
        for split in splits:
            print('============> Start extracting features for {} split'.format(split))
            start_time = timer()
            # Required directories
            audio_dir = os.path.join(data_dir, split)
            feature_dir = os.path.join(feat_dir, feature_type, audio_format, feature_description, split)
            # Empty feature folder
            shutil.rmtree(feature_dir, ignore_errors=True)
            os.makedirs(feature_dir, exist_ok=True)

            # Get audio list
            audio_fn_list = []
            for root, dir, fnames in os.walk(audio_dir, topdown=True):
                for fname in fnames:
                    if fname.endswith('.wav'):
                        fpath = os.path.join(root, fname)
                        audio_fn_list.append(fpath)
            audio_fn_list = sorted(audio_fn_list)

            # Extract features
            for count, audio_fn in enumerate(tqdm(audio_fn_list)):

                audio_input, _ = librosa.load(audio_fn, sr=fs, mono=False, dtype=np.float32)
                # Compute stft
                log_specs = []
                for imic in np.arange(n_mics):
                    stft = librosa.stft(y=np.asfortranarray(audio_input[imic, :]), n_fft=n_fft, hop_length=hop_length,
                                        center=True, window='hann', pad_mode='reflect')
                    if imic == 0:
                        n_frames = stft.shape[1]
                        X = np.zeros((n_bins, n_frames, n_mics), dtype='complex')  # (n_bins, n_frames, n_mics)
                    X[:, :, imic] = stft
                    # Compute log linear power spectrum
                    spec = (np.abs(stft) ** 2).T
                    log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
                    log_spec = np.expand_dims(log_spec, axis=0)
                    log_specs.append(log_spec)
                log_specs = np.concatenate(log_specs, axis=0)  # (n_mics, n_frames, n_bins)

                # Compute spatial feature
                phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
                if feature_type == 'salsa_ipd':
                    phase_vector = phase_vector / np.pi
                elif feature_type == 'salsa_lite':
                    phase_vector = phase_vector / (delta * freq_vector)
                phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)
                # Crop frequency
                log_specs = log_specs[:, :, lower_bin:cutoff_bin]
                phase_vector = phase_vector[:, :, lower_bin:cutoff_bin]
                phase_vector[:, :, upper_bin:] = 0

                # Stack features
                audio_feature = np.concatenate((log_specs, phase_vector), axis=0)

                # Write features to file
                audio_fname = os.path.basename(audio_fn)
                feature_fn = os.path.join(feature_dir, audio_fname.replace('wav', 'h5'))
                with h5py.File(feature_fn, 'w') as hf:
                    hf.create_dataset('feature', data=audio_feature, dtype=np.float32)
                tqdm.write('{}, {}, {}'.format(count, audio_fn, audio_feature.shape))

            print("Extracting feature finished! Elapsed time: {:.3f} s".format(timer() - start_time))

    # Compute feature mean and std for train set. For simplification, we use same mean and std for validation and
    # evaluation
    if task in ['feature_scaler', 'scaler']:
        feature_dir = os.path.join(feat_dir, feature_type, audio_format, feature_description)
        compute_scaler(feature_dir=feature_dir, audio_format=audio_format)


if __name__ == '__main__':
    extract_features()