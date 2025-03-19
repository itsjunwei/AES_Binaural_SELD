import datetime
import warnings
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from collections import defaultdict
import math

def distance_between_cartesian_coordinates_vectorized(coords1, coords2):
    """
    Vectorized angular distance between two sets of cartesian coordinates.
    coords1 and coords2 are arrays of shape (N, 3)
    """
    # Normalize the Cartesian vectors
    N1 = np.linalg.norm(coords1, axis=1, keepdims=True) + 1e-10
    N2 = np.linalg.norm(coords2, axis=1, keepdims=True) + 1e-10
    coords1_norm = coords1 / N1
    coords2_norm = coords2 / N2

    # Compute the dot product
    dot_product = np.sum(coords1_norm * coords2_norm, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angular distance in degrees
    dist = np.arccos(dot_product) * (180.0 / np.pi)
    return dist  # Shape: (N,)


def determine_similar_location_vectorized(sed_pred0, sed_pred1, doa_pred0, doa_pred1, thresh_unify):
    """
    Vectorized function to determine if two predictions are from similar locations.
    sed_pred0, sed_pred1: arrays of shape (N,)
    doa_pred0, doa_pred1: arrays of shape (N, 3)
    """
    # Active mask where both SED predictions are active
    active_mask = (sed_pred0 > 0.5) & (sed_pred1 > 0.5)  # Shape: (N,)

    # Initialize flags array
    flags = np.zeros_like(sed_pred0, dtype=bool)

    # Compute distances only where active_mask is True
    if np.any(active_mask):
        dist = distance_between_cartesian_coordinates_vectorized(
            doa_pred0[active_mask],
            doa_pred1[active_mask]
        )
        # Flags where distance is less than threshold
        flags[active_mask] = dist < thresh_unify

    return flags  # Shape: (N,)


def optimize_output_dict_creation(params, sed_pred0, sed_pred1, sed_pred2,
                                  doa_pred0, doa_pred1, doa_pred2,
                                  dist_pred0, dist_pred1, dist_pred2):
    """
    Optimized function to create output_dict without nested loops.
    Processes data per frame, vectorizing over classes.
    """
    num_frames, num_classes = sed_pred0.shape
    thresh_unify = params['thresh_unify']
    output_dict = defaultdict(list)

    for frame_idx in range(num_frames):
        # Extract data for the current frame
        sed0 = sed_pred0[frame_idx]  # Shape: (num_classes,)
        sed1 = sed_pred1[frame_idx]
        sed2 = sed_pred2[frame_idx]

        # Reshape DOA predictions to (num_classes, 3)
        doa0 = doa_pred0[frame_idx].reshape(3, num_classes).T  # Shape: (num_classes, 3)
        doa1 = doa_pred1[frame_idx].reshape(3, num_classes).T
        doa2 = doa_pred2[frame_idx].reshape(3, num_classes).T

        dist0 = dist_pred0[frame_idx]
        dist1 = dist_pred1[frame_idx]
        dist2 = dist_pred2[frame_idx]

        classes = np.arange(num_classes)

        # Determine flags
        flag_0sim1 = determine_similar_location_vectorized(sed0, sed1, doa0, doa1, thresh_unify)
        flag_1sim2 = determine_similar_location_vectorized(sed1, sed2, doa1, doa2, thresh_unify)
        flag_2sim0 = determine_similar_location_vectorized(sed2, sed0, doa2, doa0, thresh_unify)

        flag_sum = flag_0sim1.astype(int) + flag_1sim2.astype(int) + flag_2sim0.astype(int)

        # Case where flag_sum == 0
        mask_flag0 = flag_sum == 0
        mask_sed0 = mask_flag0 & (sed0 > 0.5)
        mask_sed1 = mask_flag0 & (sed1 > 0.5)
        mask_sed2 = mask_flag0 & (sed2 > 0.5)

        # Collect outputs without loops where possible
        # For sed_pred0 > 0.5
        indices = classes[mask_sed0]
        if indices.size > 0:
            output_dict[frame_idx].extend([
                [int(cls_idx), doa0[cls_idx, 0], doa0[cls_idx, 1], doa0[cls_idx, 2], dist0[cls_idx]]
                for cls_idx in indices
            ])

        # For sed_pred1 > 0.5
        indices = classes[mask_sed1]
        if indices.size > 0:
            output_dict[frame_idx].extend([
                [int(cls_idx), doa1[cls_idx, 0], doa1[cls_idx, 1], doa1[cls_idx, 2], dist1[cls_idx]]
                for cls_idx in indices
            ])

        # For sed_pred2 > 0.5
        indices = classes[mask_sed2]
        if indices.size > 0:
            output_dict[frame_idx].extend([
                [int(cls_idx), doa2[cls_idx, 0], doa2[cls_idx, 1], doa2[cls_idx, 2], dist2[cls_idx]]
                for cls_idx in indices
            ])

        # Case where flag_sum == 1
        mask_flag1 = flag_sum == 1

        # Handle flag_0sim1
        mask_flag1_0sim1 = mask_flag1 & flag_0sim1
        if np.any(mask_flag1_0sim1):
            # If sed_pred2 > 0.5
            mask_sed2 = mask_flag1_0sim1 & (sed2 > 0.5)
            indices = classes[mask_sed2]
            if indices.size > 0:
                output_dict[frame_idx].extend([
                    [int(cls_idx), doa2[cls_idx, 0], doa2[cls_idx, 1], doa2[cls_idx, 2], dist2[cls_idx]]
                    for cls_idx in indices
                ])
            # Average doa0 and doa1, dist0 and dist1
            indices = classes[mask_flag1_0sim1]
            avg_doa = (doa0[indices] + doa1[indices]) / 2
            avg_dist = (dist0[indices] + dist1[indices]) / 2
            output_dict[frame_idx].extend([
                [int(cls_idx), avg_doa[i, 0], avg_doa[i, 1], avg_doa[i, 2], avg_dist[i]]
                for i, cls_idx in enumerate(indices)
            ])

        # Handle flag_1sim2
        mask_flag1_1sim2 = mask_flag1 & flag_1sim2
        if np.any(mask_flag1_1sim2):
            # If sed_pred0 > 0.5
            mask_sed0 = mask_flag1_1sim2 & (sed0 > 0.5)
            indices = classes[mask_sed0]
            if indices.size > 0:
                output_dict[frame_idx].extend([
                    [int(cls_idx), doa0[cls_idx, 0], doa0[cls_idx, 1], doa0[cls_idx, 2], dist0[cls_idx]]
                    for cls_idx in indices
                ])
            # Average doa1 and doa2, dist1 and dist2
            indices = classes[mask_flag1_1sim2]
            avg_doa = (doa1[indices] + doa2[indices]) / 2
            avg_dist = (dist1[indices] + dist2[indices]) / 2
            output_dict[frame_idx].extend([
                [int(cls_idx), avg_doa[i, 0], avg_doa[i, 1], avg_doa[i, 2], avg_dist[i]]
                for i, cls_idx in enumerate(indices)
            ])

        # Handle flag_2sim0
        mask_flag1_2sim0 = mask_flag1 & flag_2sim0
        if np.any(mask_flag1_2sim0):
            # If sed_pred1 > 0.5
            mask_sed1 = mask_flag1_2sim0 & (sed1 > 0.5)
            indices = classes[mask_sed1]
            if indices.size > 0:
                output_dict[frame_idx].extend([
                    [int(cls_idx), doa1[cls_idx, 0], doa1[cls_idx, 1], doa1[cls_idx, 2], dist1[cls_idx]]
                    for cls_idx in indices
                ])
            # Average doa2 and doa0, dist2 and dist0
            indices = classes[mask_flag1_2sim0]
            avg_doa = (doa2[indices] + doa0[indices]) / 2
            avg_dist = (dist2[indices] + dist0[indices]) / 2
            output_dict[frame_idx].extend([
                [int(cls_idx), avg_doa[i, 0], avg_doa[i, 1], avg_doa[i, 2], avg_dist[i]]
                for i, cls_idx in enumerate(indices)
            ])

        # Case where flag_sum >= 2
        mask_flag2 = flag_sum >= 2
        if np.any(mask_flag2):
            indices = classes[mask_flag2]
            avg_doa = (doa0[indices] + doa1[indices] + doa2[indices]) / 3
            avg_dist = (dist0[indices] + dist1[indices] + dist2[indices]) / 3
            output_dict[frame_idx].extend([
                [int(cls_idx), avg_doa[i, 0], avg_doa[i, 1], avg_doa[i, 2], avg_dist[i]]
                for i, cls_idx in enumerate(indices)
            ])

    # Before returning, delete large variables that are no longer needed
    del sed_pred0, sed_pred1, sed_pred2
    del doa_pred0, doa_pred1, doa_pred2
    del dist_pred0, dist_pred1, dist_pred2

    return output_dict


def count_parameters(model):
    """Returns the total number of parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_and_print(logger, out_string):
    """Write the string to the logging file and output it simulataenously"""
    try:
        logger.write(out_string+"\n")
        logger.flush()
        print(out_string, flush=True)
    except:
        print(datetime.now().strftime("%d%m%y_%H%M%S"))


class DecayScheduler(_LRScheduler):
    """
    Decays the learning rate by a fixed factor every specified number of epochs

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        decay_factor (float, optional): Factor by which to decay the learning rate. Default is 0.98.
        min_lr (float, optional): Minimum learning rate. Default is 1e-5.
        nb_epoch_to_decay (int, optional): Number of epochs between each decay step. Default is 2.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self, optimizer, decay_factor: float = 0.98, min_lr: float = 1e-5, 
                 last_epoch: int = -1, nb_epoch_to_decay: int = 2):
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.nb_epoch_to_decay = nb_epoch_to_decay
        super(DecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch < 1:
            # Return the initial learning rates for the first epoch
            return self.base_lrs
        else:
            # Calculate how many decay steps have occurred
            decay_steps = epoch // self.nb_epoch_to_decay
            # Compute the decay factor based on the number of decay steps
            current_decay = self.decay_factor ** decay_steps
            # Apply decay to each base learning rate, ensuring it doesn't go below min_lr
            return [max(base_lr * current_decay, self.min_lr) for base_lr in self.base_lrs]


class CustomLRScheduler(_LRScheduler):
    """
    Custom learning rate scheduler with distinct phases:
    1. Warmup: Linearly increases the learning rate.
    2. Maintain: Keeps the learning rate constant.
    3. Decay: Linearly decreases the learning rate towards a minimum value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epochs (int): Total number of epochs for training.
        milestones (tuple, optional): Tuple containing two floats representing the 
                                      fractions of total_epochs for warmup and maintain phases.
                                      Default is (0.05, 0.8).
        min_lr (float, optional): Minimum learning rate after decay. Default is 1e-5.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self, optimizer, total_epochs, milestones: tuple = (0.05, 0.8), 
                 min_lr: float = 1e-5, last_epoch: int = -1):
        warmup_frac, maintain_frac = milestones
        self.warmup_epochs = int(warmup_frac * total_epochs)
        self.maintain_epochs = int(maintain_frac * total_epochs)
        self.decay_epochs = total_epochs - self.warmup_epochs - self.maintain_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Phase 1: Linear warmup
            return [
                (base_lr / self.warmup_epochs) * (epoch + 1) 
                for base_lr in self.base_lrs
            ]

        elif epoch < self.warmup_epochs + self.maintain_epochs:
            # Phase 2: Maintain the current learning rate
            return [base_lr for base_lr in self.base_lrs]

        else:
            # Phase 3: Linear decay towards min_lr
            decay_step = epoch - self.warmup_epochs - self.maintain_epochs
            decay_ratio = 1 - (decay_step / self.decay_epochs)
            return [
                base_lr * decay_ratio + self.min_lr * (1 - decay_ratio) 
                for base_lr in self.base_lrs
            ]


def warmup_cosine_annealing_lr_scheduler(optimizer, warmup_epochs: int = 5, 
                                         total_epochs: int = 100, floor: float = 1e-5):
    """
    Creates a learning rate scheduler that performs a linear warmup followed by 
    cosine annealing. The learning rate will not fall below a specified floor.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs for the linear warmup phase. Default is 5
        total_epochs (int): Total number of epochs for training. Default is 100
        floor (float, optional): Minimum learning rate. Default is 1e-5.

    Returns:
        LambdaLR: A PyTorch LambdaLR scheduler.
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Phase 1: Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Phase 2: Cosine annealing with floor
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return max(cosine_decay, floor)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CosineWarmup_StepScheduler(_LRScheduler):
    """
    Learning rate scheduler that combines linear warmup with cosine annealing based on training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_pct (float): Fraction of total steps for the linear warmup phase (e.g., 0.1 for 10%). Default is 0.05.
        total_steps (int): Total number of training steps.
        last_step (int, optional): The index of the last step. Default is -1.
    """
    
    def __init__(self, optimizer, warmup_pct: float = 0.05, 
                 total_steps: int = 10000, last_step: int = -1):
        if not 0.0 <= warmup_pct < 1.0:
            raise ValueError("warmup_pct must be in the range [0.0, 1.0).")
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        
        self.warmup_steps = int(warmup_pct * total_steps)
        self.total_steps = total_steps
        super(CosineWarmup_StepScheduler, self).__init__(optimizer, last_step)

    def get_lr(self):
        """
        Computes the learning rate for the current step.

        Returns:
            list: Updated learning rates for each parameter group.
        """
        step = self.last_epoch + 1  # Increment step count
        lr_factors = [self.get_lr_factor(step) for _ in self.base_lrs]
        return [base_lr * factor for base_lr, factor in zip(self.base_lrs, lr_factors)]

    def get_lr_factor(self, step: int) -> float:
        """
        Computes the learning rate scaling factor based on the current step.

        Args:
            step (int): Current step number.

        Returns:
            float: Scaling factor for the learning rate.
        """
        if step < self.warmup_steps and self.warmup_steps != 0:
            # Linear warmup phase
            warmup_factor = step / self.warmup_steps
        else:
            warmup_factor = 1.0

        if step <= self.total_steps:
            # Cosine annealing phase
            progress = step / self.total_steps
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            # After total_steps, keep the learning rate at the minimum
            cosine_factor = 0.0

        return warmup_factor * cosine_factor

    def step(self, step: int = None):
        """
        Updates the learning rate. Should be called after each training step.

        Args:
            step (int, optional): The current step number. If not provided, increments internally.
        """
        if step is None:
            step = self.last_epoch + 1
        else:
            if step < 0:
                raise ValueError("step must be non-negative.")
            if step > self.total_steps:
                step = self.total_steps
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler that combines linear warmup with cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup (int): Number of epochs for the linear warmup phase. Default is 5.
        max_iters (int): Total number of epochs for training. Default is 100.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self, optimizer, warmup: int = 5, 
                 max_iters: int = 100, last_epoch: int = -1):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Computes the learning rate factor based on the current epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Scaling factor for the learning rate.
        """
        # Cosine annealing factor
        cosine_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))

        if epoch <= self.warmup:
            # Linear warmup scaling
            warmup_factor = epoch / self.warmup
            return cosine_factor * warmup_factor
        else:
            return cosine_factor


class NoamLR(_LRScheduler):
    """
    Implements the Noam learning rate schedule, which increases the learning rate linearly 
    for the first `warmup_steps` and then decreases it proportionally to the inverse square root 
    of the step number.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int, optional): Number of steps to linearly increase the learning rate. Default is 10.
        dimensionality (int, optional): Dimensionality of the model, used to scale the learning rate. Default is 256.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self, optimizer, warmup_steps: int = 10, 
                 dimensionality: int = 256, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.n_dims = dimensionality
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes the scaled learning rate based on the current step.

        Returns:
            list: List of updated learning rates for each parameter group.
        """
        step = max(1, self.last_epoch)
        # Calculate the scaling factor
        scale = (self.warmup_steps ** 0.5) * min(step ** (-0.5), step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
    
def cosine_with_warmup(step, warmup_steps, total_steps, lr_min, lr_max):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))