import numpy as np

def gaussian_wavepacket(
    x: np.ndarray, x0: float = -20.0, k0: float = 3.0, sigma: float = 3.0
) -> np.ndarray:
    """Genera un pacchetto d'onda gaussiano 1D:

    psi(x) = exp(-(x - x0)^2 / (4 * sigma^2)) * exp(i * k0 * x)
    """
    gaussian = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
    plane_wave = np.exp(1j * k0 * x)
    return gaussian * plane_wave