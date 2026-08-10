import numpy as np

def harmonic_oscillator(
        x: np.ndarray, omega: float = 1.0, m:float = 1.0
) -> np.ndarray:
    """Potenziale dell'oscillatore armonico: V(x) = 0.5 * m * omega^2 * x^2"""
    return 0.5 * m * (omega**2) * (x**2)

def rectangular_barrier(
        x: np.ndarray, V0: float = 5.0, width: float = 4.0, center: float = 0.0
) -> np.ndarray:
    """Barriera di potenziale rettangolare di altezza V0 e larghezza 'width'."""
    V = np.zeros_like(x)
    mask = ( x >= center - width / 2) & (x <= center + width / 2)
    V[mask] = V0
    return V

def step_potential(
       x: np.ndarray, V0: float = 5.0, step_position: float = 0.0
) -> np.ndarray:
    """Gradino di potenziale di altezza V0."""
    V = np.zeros_like(x)
    V[x >= step_position] = V0
    return V