import numpy as np
import random

def apply_observation_noise(
    rows: list[dict],
    sigma_xy_nm: float = 0.05,
    sigma_alt_ft: float = 50.0,
    drop_prob: float = 0.1,
    seed: int | None = None,
):
    """
    apply sensor-like noise to simulated trajectory rows.
    - Gaussian noise on x/y (NM)
    - Gaussian noise on altitude (ft)
    - Randomly drop timesteps

    returns:
        noisy_rows: list[dict]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    noisy_rows = []

    for r in rows:
        # randomly drop observation
        if random.random() < drop_prob:
            continue

        noisy = r.copy()

        noisy["a_x_nm"] += np.random.normal(0.0, sigma_xy_nm)
        noisy["a_y_nm"] += np.random.normal(0.0, sigma_xy_nm)
        noisy["b_x_nm"] += np.random.normal(0.0, sigma_xy_nm)
        noisy["b_y_nm"] += np.random.normal(0.0, sigma_xy_nm)

        noisy["a_alt_ft"] += np.random.normal(0.0, sigma_alt_ft)
        noisy["b_alt_ft"] += np.random.normal(0.0, sigma_alt_ft)

        noisy_rows.append(noisy)

    return noisy_rows
