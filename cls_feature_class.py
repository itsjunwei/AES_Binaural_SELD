# Contains routines for labels creation, features extraction and normalization
import contextlib
import csv
import math
import os
import shutil
import wave

import joblib
import librosa
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from tqdm import tqdm

def create_folder(folder_name):
    """
    Create a folder if it does not exist.
    """
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def delete_and_create_folder(folder_name):
    """
    Delete a folder if it exists, and then create it.
    """
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)


def nCr(n, r):
    """
    Compute the number of combinations (n choose r).

    Parameters:
        n (int): Total number of items.
        r (int): Number of items to choose.

    Returns:
        int: The number of ways to choose `r` items from `n` items
             without repetition and without order.

    Raises:
        ValueError: If `r` is negative or greater than `n`.
    """
    if r < 0 or r > n:
        raise ValueError(f"Invalid values for n and r: n={n}, r={r}. Must satisfy 0 <= r <= n.")
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))



def compute_msc_recursive(X, lambda_=0.8):
    """
    Compute the Mean Magnitude-Squared Coherence (MMSC) using recursive time averaging.

    The MMSC provides a smoothed estimate of the MSC over time, leveraging a forgetting factor
    to give more weight to recent frames.

    Parameters:
        X (ndarray): Multichannel time-frequency representation of shape (M, T, F).
        lambda_ (float, optional): Forgetting factor for recursive averaging, 0 < lambda_ < 1.

    Returns:
        gamma_avg (ndarray): Mean magnitude-squared coherence averaged over all channel pairs
                             for each TF bin, shape (T, F).

    Raises:
        ValueError: If input `X` does not have three dimensions or `lambda_` is not in (0, 1).
    """
    if X.ndim != 3:
        raise ValueError(f"Input X must be a 3D array of shape (M, T, F), but got shape {X.shape}.")
    if not (0 < lambda_ < 1):
        raise ValueError(f"Parameter lambda_ must be in the interval (0, 1), but got lambda_={lambda_}.")

    M, T, F = X.shape
    num_pairs = M * (M - 1) // 2
    gamma_avg = np.zeros((T, F), dtype=np.float32)

    # Generate indices for all unique microphone pairs (i < j)
    i_indices, j_indices = np.triu_indices(M, k=1)

    # Initialize recursive variables for auto-power and cross-power spectral densities
    S_ii = np.zeros((M, F), dtype=np.complex64)       # Auto-power spectral densities
    S_ij = np.zeros((num_pairs, F), dtype=np.complex64)  # Cross-power spectral densities

    # Iterate over each time frame to compute recursive averages
    for t in range(T):
        # Extract the current time frame across all channels and frequencies
        X_t = X[:, t, :]            # Shape: (M, F)
        X_t_conj = np.conj(X_t)     # Shape: (M, F)

        # Update auto-power spectral densities with recursive averaging
        S_ii = lambda_ * S_ii + (1 - lambda_) * (X_t * X_t_conj)

        # Update cross-power spectral densities for each microphone pair with recursive averaging
        Xi = X_t[i_indices, :]        # Shape: (num_pairs, F)
        Xj_conj = X_t_conj[j_indices, :]  # Shape: (num_pairs, F)
        S_ij = lambda_ * S_ij + (1 - lambda_) * (Xi * Xj_conj)

        # Compute the numerator and denominator for MSC
        numerator = np.abs(S_ij) ** 2
        denominator = S_ii[i_indices, :] * S_ii[j_indices, :]

        # Compute MSC for each microphone pair and frequency bin
        gamma_ij = numerator / denominator

        # Average MSC across all microphone pairs for the current time frame
        gamma_avg[t, :] = np.mean(gamma_ij.real, axis=0)

    return gamma_avg


class FeatureClass:
    """
    Contains routines for label creation, feature extraction, and normalization.
    """

    def __init__(self, params, is_eval=False):
        """
        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """
        self._parameters = params
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')
        self._vid_dir = os.path.join(self._dataset_dir, 'video_{}'.format('eval' if is_eval else 'dev'))

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None
        self._vid_feat_dir = None
        self._upper_bin = None

        # Local parameters
        self._is_eval = is_eval
        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        # Determining NFFT points
        if self._hop_len == 300:
            self._win_len = 512
            self._nfft = 512
        else:
            self._win_len = 2 * self._hop_len
            self._nfft = self._next_greater_power_of_2(self._win_len)
        print("Using {}-point FFT".format(self._nfft))

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = params['nb_channels']
        self._normalized_distances = params['normalize_distance']
        if self._normalized_distances:
            self._dmean = params['dmean']
            self._dstd = params['dstd']
            self._dmax = params['dmax']

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']

        # SALSA-Lite configurations
        if self._use_salsalite:
            # Initialize the spatial feature constants
            self._lower_bin = int(np.floor(params["fmin_doa_salsalite"] * self._nfft / float(self._fs)))
            self._lower_bin = max(1, self._lower_bin)
            self._upper_bin = int(
                np.floor(np.min((params["fmax_doa_salsalite"], self._fs // 2)) * self._nfft / float(self._fs))
            )

            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft // 2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]

            # Initialize spectral feature constants
            self._cutoff_bin = int(np.floor(params["fmax_spectra_salsalite"] * self._nfft / float(self._fs))) # 192
            assert (
                self._upper_bin <= self._cutoff_bin
            ), f"Upper bin for doa feature {self._upper_bin} is higher than cutoff bin {self._cutoff_bin}!"
            self._nb_mel_bins = self._cutoff_bin - self._lower_bin

            print("SALSA Lite Configurations")
            print(
                f"\tBin Range (Lower - Upper - Nb Melbins): [{self._lower_bin}, "
                f"{self._upper_bin}, {self._nb_mel_bins}]\n"
            )

        else:
            self._nb_mel_bins = params["nb_mel_bins"]
            self._mel_wts = librosa.filters.mel(
                sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins
            ).T

        # Sound event classes dictionary
        self._nb_unique_classes = params["unique_classes"]
        self._filewise_frames = {}

    def get_frame_stats(self):
        """
        Compute frame statistics for all audio files. Fills `self._filewise_frames`.
        """
        if len(self._filewise_frames) != 0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename), 'r')) as f:
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def _load_audio(self, audio_path):
        """
        Load audio from file using scipy wav.
        """
        fs, audio = wav.read(audio_path)
        audio = audio[:, : self._nb_channels] / 32768.0 + self._eps
        return audio, fs


    @staticmethod
    def _next_greater_power_of_2(x):
        """
        Return the next power of two for integer x.
        """
        return 2 ** ((x - 1).bit_length())

    def _binaural_features(self, audio_input, _nb_frames):
        """
        Compute a feature stack from a binaural audio input, concatenating:
        - Mean magnitude spectrogram
        - Sine of interchannel phase differences
        - Cosine of interchannel phase differences
        - Interchannel level differences (in dB)

        The returned array has shape (4, time, frequency), where the four channels are in the order listed.

        Assumes:
        audio_input: numpy array of shape (samples, 2)
        _nb_frames: number of time frames to consider from the STFT
        """

        # Check that the input is indeed binaural
        if audio_input.shape[1] != 2:
            raise ValueError("Input audio must be binaural (2 channels)")

        # Compute STFT for each channel
        stft_left = librosa.core.stft(
            np.asfortranarray(audio_input[:, 0]),
            n_fft=self._nfft,
            hop_length=self._hop_len,
            win_length=self._win_len,
            window="hann",
        )[1:, :_nb_frames] # Remove the DC component

        stft_right = librosa.core.stft(
            np.asfortranarray(audio_input[:, 1]),
            n_fft=self._nfft,
            hop_length=self._hop_len,
            win_length=self._win_len,
            window="hann",
        )[1:, :_nb_frames] # Remove the DC component

        # Compute magnitude spectrograms
        mag_left = np.abs(stft_left)
        mag_right = np.abs(stft_right)

        # Mean magnitude spectrogram (transposed to (time, freq))
        mean_mag = ((mag_left + mag_right) / 2).T

        # Phase differences between channels
        phase_left = np.angle(stft_left)
        phase_right = np.angle(stft_right)
        phase_diff = phase_left - phase_right

        # Sine and cosine of the phase differences (transposed to (time, freq))
        sin_phase = np.sin(phase_diff).T
        cos_phase = np.cos(phase_diff).T

        # Interchannel level differences (in dB)
        # Adding a small epsilon to avoid taking log of zero.
        eps = 1e-8
        ild = (20 * np.log10(mag_left + eps) - 20 * np.log10(mag_right + eps)).T

        # Stack features along a new axis: (channels, time, frequency)
        features = np.stack([mean_mag, sin_phase, cos_phase, ild], axis=0)
        return features


    def _spectrogram(self, audio_input, _nb_frames):
        """
        Compute STFT spectrogram for each channel of the audio input.
        """
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(
                np.asfortranarray(audio_input[:, ch_cnt]),
                n_fft=self._nfft,
                hop_length=self._hop_len,
                win_length=self._win_len,
                window="hann",
            )
            spectra.append(stft_ch[:, :_nb_frames])  # (freq x time)

        return np.array(spectra).T  # (time x freq x channels)


    def _get_mel_spectrogram(self, linear_spectra):
        """
        Compute log-mel spectrogram from linear spectrogram for each channel.
        """
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1))  # (time, channel, freq)
        mel_feat = mel_feat.reshape((linear_spectra.shape[0], -1))  # (time, channel * freq)
        return mel_feat


    def _get_binauralfeat_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        features = self._binaural_features(audio_in, nb_feat_frames)
        return features


    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec


    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    #print(active_event)
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
                    dist_label[frame_ind, active_event[0]] = active_event[5]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label, dist_label), axis=1)
        return label_mat

    # OUTPUT LABELS
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 0, active_event_a0[0]] = (active_event_a0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 1, active_event_b0[0]] = (active_event_b0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 2, active_event_b1[0]] = (active_event_b1[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 3, active_event_c0[0]] = (active_event_c0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 4, active_event_c1[0]] = (active_event_c1[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 5, active_event_c2[0]] = (active_event_c2[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 0, active_event_a0[0]] = (active_event_a0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 1, active_event_b0[0]] = (active_event_b0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 2, active_event_b1[0]] = (active_event_b1[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 3, active_event_c0[0]] = (active_event_c0[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 4, active_event_c1[0]] = (active_event_c1[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            if self._normalized_distances:
                                dist_label[frame_ind, 5, active_event_c2[0]] = (active_event_c2[5]/100. - self._dmean)/(self._dstd * self._dmax)
                            else:
                                dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)  # [nb_frames, 6, 5(=act+XYZ+dist), max_classes]
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_file_feature(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in

        feat = self._get_binauralfeat_for_file(_wav_path)

        print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feat.shape))
        np.save(_feat_path, feat)


    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)
        from multiprocessing import Pool
        import time
        start_s = time.time()
        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        arg_list = []
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                wav_path = os.path.join(loc_aud_folder, wav_filename)
                feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                self.extract_file_feature((file_cnt, wav_path, feat_path))
                arg_list.append((file_cnt, wav_path, feat_path))
        print(time.time()-start_s)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # Pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print(f'\t\tfeat_dir: {self._feat_dir}')

            # Initialize spec_scalers as None; it will be initialized after loading the first file
            spec_scalers = None
            spec_scaler = preprocessing.StandardScaler()

            # Iterate through all feature files to fit the scalers
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print(f'{file_cnt}: {file_name}')
                feat_path = os.path.join(self._feat_dir, file_name)
                feat_file = np.load(feat_path)  # Expected shape: (channels, time, frequency)

                if feat_file.ndim == 3:

                    if spec_scalers is None:
                        num_channels = feat_file.shape[0]
                        spec_scalers = [preprocessing.StandardScaler() for _ in range(num_channels)]
                        print(f'Initialized {num_channels} StandardScalers for each channel.')

                    # Fit each scaler with the data from its respective channel
                    for ch in range(num_channels):
                        # Reshape the data to (samples, features) if necessary
                        # Here, treating 'time' as samples and 'frequency' as features
                        channel_data = feat_file[ch].reshape(-1, feat_file.shape[2])  # Shape: (time, frequency)
                        spec_scalers[ch].partial_fit(channel_data)

                elif feat_file.ndim == 2:
                    spec_scaler.partial_fit(feat_file)

                else:
                    raise ValueError("Feature is of unknown dimensions : {}".format(feat_file.shape))

                # Clean up
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )

            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print(f'\t\tfeat_dir_norm: {self._feat_dir_norm}')

        # Iterate through all feature files to apply normalization
        for file_cnt, file_name in tqdm(enumerate(os.listdir(self._feat_dir)), total=len(os.listdir(self._feat_dir)), desc="Transforming features..."):
            feat_path = os.path.join(self._feat_dir, file_name)
            feat_file = np.load(feat_path)  # Shape: (channels, time, frequency)
            
            if feat_file.ndim == 3:

                if spec_scalers is None:
                    raise ValueError("spec_scalers not initialized. Ensure that the scalers are properly loaded or fitted.")

                # Apply the scaler for each channel
                for ch in range(num_channels):
                    # Reshape the data to (samples, features) if necessary
                    channel_data = feat_file[ch].reshape(-1, feat_file.shape[2])  # Shape: (time, frequency)
                    normalized_channel = spec_scalers[ch].transform(channel_data)
                    feat_file[ch] = normalized_channel.reshape(feat_file.shape[1], feat_file.shape[2])

                # Save the normalized feature file
                normalized_feat_path = os.path.join(self._feat_dir_norm, file_name)
                if "fold4" in normalized_feat_path: # Validation data loader needs this format
                    feat_file = np.transpose(feat_file, (1, 0, 2)) # CTF -> TCF
                    feat_file = feat_file.reshape(feat_file.shape[0], -1) # TCF -> T, C*F
                    np.save(normalized_feat_path, feat_file)
                else: # Otherwise self data-loader is ndim robust
                    np.save(normalized_feat_path, feat_file)

            elif feat_file.ndim == 2:
                feat_file = spec_scaler.transform(feat_file)

                np.save(
                    os.path.join(self._feat_dir_norm, file_name),
                    feat_file
                )

            else:
                raise ValueError("Feature is of unknown dimensions : {}".format(feat_file.shape))

            # Clean up
            del feat_file

        print(f'Normalized files written to {self._feat_dir_norm}')

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)
        for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                if self._multi_accdoa:
                    label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                else:
                    label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file, cm2m=False):  # TODO: Reconsider cm2m conversion
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        _words = []     # For empty files
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:  # frame, class idx,  polar coordinates(2) # no distance data, for example in eval pred
                _output_dict[_frame_ind].append([int(_words[1]), 0, float(_words[2]), float(_words[3])])
            if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])/100 if cm2m else float(_words[5])])
            elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])/100 if cm2m else float(_words[6])])
        _fid.close()
        if len(_words) == 7:
            _output_dict = self.convert_output_format_cartesian_to_polar(_output_dict)
        return _output_dict


    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
                # TODO: What if our system estimates track cound and distence (or only one of them)
        _fid.close()


    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict


    def organize_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in every frame, similar to segment_labels but at frame level
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each frame
                dictionary_name[frame-index][class-index][track-index] = [azimuth, elevation, (distance)] or
                                                                         [x, y, z, (distance)]
        '''
        nb_frames = _max_frames
        output_dict = {x: {} for x in range(nb_frames)}
        for frame_idx in range(0, _max_frames):
            if frame_idx not in _pred_dict:
                continue
            for [class_idx, track_idx, *localization] in _pred_dict[frame_idx]:
                if class_idx not in output_dict[frame_idx]:
                    output_dict[frame_idx][class_idx] = {}
                
                if track_idx not in output_dict[frame_idx][class_idx]:
                    output_dict[frame_idx][class_idx][track_idx] = localization
                else:
                    # Repeated track_idx for the same class_idx in the same frame_idx, the model is not estimating
                    # track IDs, so track_idx is set to a negative value to distinguish it from a proper track ID
                    min_track_idx = np.min(np.array(list(output_dict[frame_idx][class_idx].keys())))
                    new_track_idx = min_track_idx - 1 if min_track_idx < 0 else -1
                    output_dict[frame_idx][class_idx][new_track_idx] = localization

        return output_dict


    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict


    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180.

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [x, y, z] + tmp_val[4:])
        return out_dict


    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [azimuth, elevation] + tmp_val[5:])
        return out_dict


    # ------------------------------- Misc public functions -------------------------------

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
               '{}_label'.format('{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
        )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_vid_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'video_{}'.format('eval' if self._is_eval else 'dev'))

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins