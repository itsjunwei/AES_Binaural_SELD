"""
@ Tho Nguyen, NTU, 2021 04 07
This module includes code to do data augmentation in STFT domain on numpy array:
    1. random volume
    2. random cutout
    3. spec augment
    4. freq shift
==================================================
Example how to use data augmentation
# import
from transforms import CompositeCutout, ComposeTransformNp, RandomShiftUpDownNp, RandomVolumeNp
# call transform
train_transform = ComposeTransformNp([
    RandomShiftUpDownNp(freq_shift_range=10),
    RandomVolumeNp(),
    CompositeCutout(image_aspect_ratio=320 / 128),  # 320: number of frames, 128: n_mels
    ])
# perform data augmentation
X = train_transform(X)  # X size: 1 x n_frames x n_mels
"""
import numpy as np
import random

class RandomSpecAugHole:
    """
    Apply random hole masking to the spectrogram.
    """
    def __init__(self, num_holes=5, hole_height=10, hole_width=10, p=0.8):
        """
        :param num_holes: Number of random holes to apply.
        :param hole_height: Maximum height of each hole in frequency bins.
        :param hole_width: Maximum width of each hole in time steps.
        :param p: Probability of applying the SpecAugHole augmentations technique
        """
        self.num_holes = num_holes
        self.hole_height = hole_height
        self.hole_width = hole_width
        self.p = p

    def __call__(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): Spectrogram of shape (channels, time, frequency).
        Returns:
            Tensor: Augmented spectrogram with random holes.
        """
        if np.random.rand() > self.p:
                return spectrogram
        else:
            cloned = spectrogram.clone()
            channels, time_steps, freq_bins = cloned.size()
            n_holes = random.randint(1, self.num_holes)

            for _ in range(n_holes):
                # Randomly choose hole size
                height = random.randint(1, self.hole_height)
                width = random.randint(1, self.hole_width)

                # Randomly choose top-left corner of the hole
                freq_start = random.randint(0, max(1, freq_bins - height))
                time_start = random.randint(0, max(1, time_steps - width))

                # Apply the mask
                cloned[:, time_start:time_start + width, freq_start:freq_start + height] = 0

            return cloned


class SpecAugment:
    def __init__(self, time_masking=2, freq_masking=2, time_mask_param=2, freq_mask_param=2):
        """
        Args:
            time_masking (int): Number of time masks to apply.
            freq_masking (int): Number of frequency masks to apply.
            time_mask_param (int): Maximum width of each time mask.
            freq_mask_param (int): Maximum width of each frequency mask.
        """
        self.time_masking = time_masking
        self.freq_masking = freq_masking
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, X):
        # Apply time masks
        for _ in range(self.time_masking):
            X = self._time_mask(X, self.time_mask_param)
        
        # Apply frequency masks
        for _ in range(self.freq_masking):
            X = self._freq_mask(X, self.freq_mask_param)

        return X

    def _time_mask(self, X, max_width):
        _, time, _ = X.shape
        t = random.randint(0, time - 1)
        width = random.randint(0, min(max_width, time - t))
        X[:, t:t + width, :] = 0
        return X

    def _freq_mask(self, X, max_width):
        _, _, freq = X.shape
        f = random.randint(0, freq - 1)
        width = random.randint(0, min(max_width, freq - f))
        X[:, :, f:f + width] = 0
        return X

def freq_mixup(data, n_swap=None):
    """Implementing the FreqMix augmentation technique.
    
    Assuming the data is of shape (channels, timesteps, freq bins)
    
    Input
        data : Spectrogram data (ch, time, freq)
        n_swap : Number of frequency bins to swap. Defaults to None, which will swap 8% of frequency bins
        
    Returns
        data : Same spectrogram with frequency bins mixed up (ch, time, freq)"""
        
    n_ch, n_time, n_freq = data.shape
    
    if n_swap is None:
        n_swap = int(n_freq * 0.08)
        
    assert n_swap/n_freq < 0.2, "Try not to swap more than 20 percent of the frequency bins at once"

    x = data.copy()
    
    f_0 = np.random.randint(0, int(n_freq - 2*n_swap))
    f_1 = np.random.randint(int(f_0 + n_swap), int(n_freq - n_swap))
    
    f_low = data[:, :, f_0:f_0+n_swap]
    f_hi  = data[:, :, f_1:f_1+n_swap]

    x[:, :, f_0:f_0+n_swap] = f_hi
    x[:, :, f_1:f_1+n_swap] = f_low
    
    return x


def time_mixup(data, target,
               low_lim = 10, upp_lim = 60):
    """Function that implements mixup in the time domain.
    Assumes that the data is in the shape of (channels, timesteps, frequencies)
    Target labels in the shape of (timesteps, ...)

    Input
        data : Array of 2 data values
        target : Array of 2 corresponding target values
        low_lim : Lower limit of the mixup portion in percentage
        upp_lim : Upper limit of the mixup portion in percentage

    Returns
        mix_data_1, mix_data_2 : Mixed up data values
        mix_target_1, mix_target_2 : Mixed up corresponding target values"""
        
    x = np.copy(data)
    y = np.copy(target)

    # Taking the i, i+1 data and target samples
    d_0 = x[0]
    d_1 = x[1]
    d_time = d_0.shape[1] # Data timesteps

    t_0 = y[0]
    t_1 = y[1]
    t_time = t_0.shape[0] # Label timesteps

    # Getting the feature downsample rate
    time_downsample = int(d_time/t_time)

    # Generate a random float value of [0.10, 0.60)
    lam = np.random.randint(low_lim,upp_lim) / 100

    # Determining the index value for the data, target timesteps
    t_index = int(np.floor(lam * t_time))
    d_index = t_index * time_downsample

    # Getting the front and back data and target segments
    d_01 = d_0[:, :d_index, :]
    d_02 = d_0[:, d_index:, :]

    d_11 = d_1[:, :d_index, :]
    d_12 = d_1[:, d_index:, :]

    t_01 = t_0[:t_index, :, :, :]
    t_02 = t_0[t_index:, :, :, :]

    t_11 = t_1[:t_index, :, :, :]
    t_12 = t_1[t_index:, :, :, :]

    # Now we combine the segmented parts
    mix_data_1 = np.concatenate((d_01, d_12), axis=1)
    mix_target_1 = np.concatenate((t_01, t_12), axis=0)

    mix_data_2 = np.concatenate((d_11, d_02), axis=1)
    mix_target_2 = np.concatenate((t_11, t_02), axis=0)

    return mix_data_1, mix_data_2, mix_target_1, mix_target_2

def tf_mixup(data, target, use_freq=False, freq_p = 0.5, n_freq_mix = None,
             use_time = False, time_p = 0.5, t_low=10, t_hi=60):
    """
    Implements the Time-Frequency Mixup data augmentation technique. 
    
    Inputs
        data : Array of 2 data values
        target : Array of 2 corresponding target values
        use_freq (boolean) : True if use Freq Mixup
        freq_p (float) : Probability of using frequency mixup
        n_freq_mix (int) : How many frequency bins to mixup. None if wish to default to mix only 8% of frequency bins
        use_time (boolean) : True if use Time Mixup
        low_lim (int) : Lower limit of the mixup portion in percentage
        upp_lim (int) : Upper limit of the mixup portion in percentage

    Returns
        mix_data_1, mix_data_2 : Mixed up data values
        mix_target_1, mix_target_2 : Mixed up corresponding target values
    """

    x = np.copy(data)
    y = np.copy(target)

    if use_time:
        if np.random.rand() < time_p:
            d1, d2, t1, t2 = time_mixup(x, y,
                                        low_lim=t_low, upp_lim=t_hi)
            data[0] = d1
            data[1] = d2
            target[0] = t1
            target[1] = t2

    if use_freq:
        for idx in range(len(data)):
            if np.random.rand() < freq_p:
                data[idx] = freq_mixup(data[idx], n_swap=n_freq_mix)
    
    return data[0], data[1], target[0], target[1]

class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x


class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError


class RandomCutoutNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features) or (n_time_steps, n_features)>: input spectrogram.
        :return: random cutout x
        """
        # get image size
        image_dim = x.ndim
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        # Initialize output
        output_img = x.copy()
        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            if self.n_zero_channels is None:
                output_img[:, top:top + h, left:left + w] = c
            else:
                output_img[:-self.n_zero_channels,  top:top + h, left:left + w] = c
                if self.is_filled_last_channels:
                    output_img[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return output_img


class SpecAugmentNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[1]
        n_freqs = x.shape[2]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, start_idx:start_idx + dur, :] = random_value
            else:
                new_spec[:-self.n_zero_channels, start_idx:start_idx + dur, :] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, start_idx:start_idx + dur, :] = 0.0

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, :, start_idx:start_idx + dur] = random_value
            else:
                new_spec[:-self.n_zero_channels, :, start_idx:start_idx + dur] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, :, start_idx:start_idx + dur] = 0.0

        return new_spec


class RandomCutoutHoleNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                 is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
        n_cutout_holes = self.n_max_holes
        for ihole in np.arange(n_cutout_holes):
            # w = np.random.randint(4, self.max_w_size, 1)[0]
            # h = np.random.randint(4, self.max_h_size, 1)[0]
            w = self.max_w_size
            h = self.max_h_size
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.filled_value is None:
                filled_value = np.random.uniform(min_value, max_value)
            else:
                filled_value = self.filled_value
            if self.n_zero_channels is None:
                new_spec[:, top:top + h, left:left + w] = filled_value
            else:
                new_spec[:-self.n_zero_channels, top:top + h, left:left + w] = filled_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return new_spec


class CompositeCutout(DataAugmentNumpyBase):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio,
                                            n_zero_channels=n_zero_channels,
                                            is_filled_last_channels=is_filled_last_channels)
        self.spec_augment = SpecAugmentNp(always_apply=True, n_zero_channels=n_zero_channels,
                                          is_filled_last_channels=is_filled_last_channels)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True, n_zero_channels=n_zero_channels,
                                                     is_filled_last_channels=is_filled_last_channels)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 3, 1)[0]
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout_hole(x)


class RandomShiftUpDownNp(DataAugmentNumpyBase):
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect',
                 n_last_channels: int = 0):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels

    def apply(self, x: np.ndarray):
        if self.always_apply is False:
            return x
        else:
            if np.random.rand() < self.p:
                return x
            else:
                n_channels, n_timesteps, n_features = x.shape
                if self.freq_shift_range is None:
                    self.freq_shift_range = int(n_features * 0.08)
                shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
                if self.direction is None:
                    direction = np.random.choice(['up', 'down'], 1)[0]
                else:
                    direction = self.direction
                new_spec = x.copy()
                if self.n_last_channels == 0:
                    if direction == 'up':
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec = np.pad(new_spec, ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                else:
                    if direction == 'up':
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
                    else:
                        new_spec[:-self.n_last_channels] = np.pad(
                            new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
                return new_spec


#############################################################################
# Joint transform
class ComposeMapTransform:
    """
    Compose a list of data augmentation on numpy array. These data augmentation methods change both features and labels.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        for transform in self.transforms:
            x, y_sed, y_doa = transform(x, y_sed, y_doa)
        return x, y_sed, y_doa


class MapDataAugmentBase:
    """
    Base class for joint feature and label augmentation.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        if self.always_apply:
            return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
        else:
            if np.random.rand() < self.p:
                return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
            else:
                return x, y_sed, y_doa

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x: < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_sed: <np.ndarray (n_time_steps, n_classes)>
        :param y_doa: <np.ndarray (n_time_steps, 3*nclasses)>
        n_channels = 7 for salsa, melspeciv, linspeciv; 10 for melspecgcc, linspecgcc
        """
        raise NotImplementedError


class TfmapRandomSwapChannelFoa(MapDataAugmentBase):
    """
    This data augmentation random swap xyz channel of tfmap of FOA format. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, accdoa
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa
        This data augmentation change x_sed and y_doa
        x feature: x[-3]: Y, x[-2]: Z, x[-1]: X
            W Y Z X Y Z X: 7 channels
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 7, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(4,))
        # change input feature
        if m[0] == 1:  # random swap x, y
            x_new[1] = x[3]
            x_new[3] = x[1]
            x_new[-3] = x[-1]
            x_new[-1] = x[-3]
        if m[1] == 1:  # negate x
            x_new[-1] = -x_new[-1]
        if m[2] == 1:  # negate y
            x_new[-3] = -x_new[-3]
        if m[3] == 1:  # negate z
            x_new[-2] = -x_new[-2]
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1:
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, 0:self.n_classes]
            if m[2] == 1:
                y_doa_new[:, self.n_classes: 2*self.n_classes] = - y_doa_new[:, self.n_classes: 2*self.n_classes]
            if m[3] == 1:
                y_doa_new[:, 2*self.n_classes:] = - y_doa_new[:, 2*self.n_classes:]
        else:
            raise NotImplementedError('this output format not yet implemented')

        return x_new, y_sed, y_doa_new


class TfmapRandomSwapChannelMic(MapDataAugmentBase):
    """
    This data augmentation random swap channels of tfmap of MIC format.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, accdoa
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa, reg_accdoa
        This data augmentation change x and y_doa
        x: x[0]: M1, x[1] = M2, x[2]: M3, x[3]: M4
            M1 M2 M3 M4 p12 p13 p14: 7 channels
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 7, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(3,))
        # change inpute feature
        if m[0] == 1:  # swap M2 and M3 -> swap x and y
            x_new[1] = x[2]
            x_new[2] = x[1]
            x_new[-3] = x[-2]
            x_new[-2] = x[-3]
        if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
            x_cur = x_new.copy()
            x_new[0] = x_cur[3]
            x_new[3] = x_cur[0]
            x_new[-1] = - x_cur[-1]
            x_new[-2] = x_cur[-2] - x_cur[-1]
            x_new[-3] = x_cur[-3] - x_cur[-1]
        if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
            x_cur = x_new.copy()
            x_new[0] = x_cur[1]
            x_new[1] = x_cur[0]
            x_new[2] = x_cur[3]
            x_new[3] = x_cur[2]
            x_new[-3] = - x_cur[-3]
            x_new[-2] = x_cur[-1] - x_cur[-3]
            x_new[-1] = x_cur[-2] - x_cur[-3]
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1:  # swap M2 and M3 -> swap x and y
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
                temp = - y_doa_new[:, 0:self.n_classes].copy()
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, self.n_classes:2 * self.n_classes] = temp
            if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
                y_doa_new[:, self.n_classes:2 * self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, 2 * self.n_classes:] = - y_doa_new[:, 2 * self.n_classes:]
        else:
            raise NotImplementedError('this doa format not yet implemented')

        return x_new, y_sed, y_doa_new


class GccRandomSwapChannelMic(MapDataAugmentBase):
    """
    This data augmentation random swap channels of melspecgcc or linspecgcc of MIC format.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, reg_polar, accdoa, reg_accdoa, cla_polar
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa, reg_accdoa
        This data augmentation change x and y_doa
        x: x[0]: M1, x[1] = M2, x[2]: M3, x[3]: M4
            M1 M2 M3 M4 xc12 xc13 xc14 xc23 xc24 xc34: 10 channels
        M1: n_timesteps x n_mels
        xc12: n_timesteps x n_lags (n_mels = n_lags)
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 10, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(3,))
        if m[0] == 1:  # swap M2 and M3 -> swap x and y
            x_new[1] = x[2]
            x_new[2] = x[1]
            x_new[4] = x[5]
            x_new[5] = x[4]
            x_new[7] = np.flip(x[7], axis=-1)
            x_new[-1] = x[-2]
            x_new[-2] = x[-1]
        elif m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
            x_cur = x_new.copy()
            x_new[0] = x_cur[3]
            x_new[3] = x_cur[0]
            x_new[4] = np.flip(x_cur[8], axis=-1)
            x_new[5] = np.flip(x_cur[9], axis=-1)
            x_new[6] = np.flip(x_cur[6], axis=-1)
            x_new[8] = np.flip(x_cur[4], axis=-1)
            x_new[9] = np.flip(x_cur[5], axis=-1)
        elif m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
            x_cur = x_new.copy()
            x_new[0] = x_cur[1]
            x_new[1] = x_cur[0]
            x_new[2] = x_cur[3]
            x_new[3] = x_cur[2]
            x_new[4] = np.flip(x_cur[4], axis=-1)
            x_new[5] = x_cur[8]
            x_new[6] = x_cur[7]
            x_new[7] = x_cur[6]
            x_new[8] = x_cur[5]
            x_new[9] = np.flip(x_cur[9], axis=-1)
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1: # swap M2 and M3 -> swap x and y
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
                temp = - y_doa_new[:, 0:self.n_classes].copy()
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, self.n_classes:2 * self.n_classes] = temp
            if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
                y_doa_new[:, self.n_classes:2 * self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, 2 * self.n_classes:] = - y_doa_new[:, 2 * self.n_classes:]
        else:
            raise NotImplementedError('this doa format not yet implemented')

        return x_new, y_sed, y_doa_new

if __name__ == "__main__":
    # Below is just my internal testing for the audio mixup function
    data = np.random.randn(32, 7, 400, 64)
    target = np.random.randn(32, 100, 6, 5, 13)

    a = data[:2]
    b = target[:2]

    d = [data[0], data[1]]
    t = [target[0], target[1]]
    d = np.array(d)
    d0 = np.copy(d)
    t = np.array(t)
    t0 = np.copy(t)
    
    print("Before \t {} , {}".format(d.shape, t.shape))
    
    d1, d2, t1, t2 = tf_mixup(d0, t0, use_freq=True, freq_p=1.0,
                    use_time=False, time_p=1.0)
    
    print("After \t {} , {}".format(d.shape, t.shape))
    print(np.array_equal(a,d1), np.array_equal(b,t1))
    
    # m1, m2, m3, m4 = time_mixup(d,t)
    # print(m1.shape, m2.shape)
    # print(m3.shape, m4.shape)
    
    # for i in range(10000):
    #     x = freq_mixup(data[0])