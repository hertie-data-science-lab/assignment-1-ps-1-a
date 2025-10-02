import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    This function implements the following cosine annealing learning rate schedule: 
    
    l_t = l_T + ((l_0 - l_T) / 2) * (1 + cos(pi * t / T))

      - l_0: initial_lr
      - l_T: min_lr (final lr)
      - t: epoch (current iteration)
      - T: total_epochs (total iterations)

    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        min_lr (float): Minimum learning rate to reach.
        
    Returns:
        float: Adjusted learning rate for the current epoch.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    # In order to avoid negative values in the formula
    t = max(0, min(epoch, total_epochs))
    cos_inner = np.pi * t / total_epochs
    return float(min_lr + (initial_lr - min_lr) * 0.5 * (1.0 + np.cos(cos_inner)))