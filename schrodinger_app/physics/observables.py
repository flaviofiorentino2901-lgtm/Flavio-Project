import numpy as np
from core.state import QuantumState

def norm(state: QuantumState) -> float:
    """Calcola la norma globale: \int |psi|^2 dx"""
    return float(np.sum(state.probability_density) * state.dx)

def expected_position(state: QuantumState) -> float:
    """Calcola il valore d'aspettazione della posizione <x> = \int x |psi|^2 dx"""
    return float(np.sum(state.x * state.probability_density) * state.dx)

def expected_momentum(state: QuantumState) -> float:
    """Calcola il valore d'aspettazione dell'impulso <p> nello spazio p."""
    psi_p = np.fft.fft(state.psi)
    prob_p = np.abs(psi_p) ** 2

    # Normalizzazione nello spazio dei momenti (Parseval)
    prob_p_norm = prob_p / np.sum(prob_p)
    return float(np.sum(state.p * prob_p_norm))