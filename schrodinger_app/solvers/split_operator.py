import numpy as np
from core.state import QuantumState
from solvers.base_solver import BaseSolver


class SplitStepSolver(BaseSolver):
    """Integratore temporale per l'equazione di Schrödinger 1D

    basato sull'algoritmo Split-Step Fourier.
    """

    def __init__(self, state: QuantumState, dt: float = 0.05):
        # Inizializza la classe base (imposta self.state e self.dt)
        super().__init__(state, dt)

        # Pre-calcoliamo gli operatori d'evoluzione temporale
        # 1. Operatore Potenziale nello spazio delle posizioni (mezzo step: dt / 2)
        self.exp_V = np.exp(-1j * (self.state.V / self.state.hbar) * (self.dt / 2.0))

        # 2. Operatore Cinetico nello spazio dei momenti (step intero: dt)
        self.exp_T = np.exp(
            -1j
            * ((self.state.p**2) / (2.0 * self.state.m * self.state.hbar))
            * self.dt
        )

    def step(self):
        """Avanza lo stato quantistico di un intervallo temporale dt."""
        # 1. Mezzo step nel potenziale (Spazio x)
        self.state.psi *= self.exp_V

        # 2. Passaggio allo spazio dei momenti tramite FFT
        psi_p = np.fft.fft(self.state.psi)

        # 3. Step intero nell'energia cinetica (Spazio p)
        psi_p *= self.exp_T

        # 4. Ritorno allo spazio delle posizioni tramite IFFT
        self.state.psi = np.fft.ifft(psi_p)

        # 5. Mezzo step nel potenziale (Spazio x)
        self.state.psi *= self.exp_V