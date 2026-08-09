import numpy as np
from core.state import QuantumState

class SplitStepSolver:
    """Use Split-Step fourier method to solve the  time dependent 1D Schroedinger equation"""
    def __init__(self,state: QuantumState, dt: float = 0.05):
        self.state = state
        self.dt = dt 
        # pre compute the time evolution operators in order not to compute htem at every time step
        self._update_operators()

    def _update_operators(self):

        """Calcola gli operatori di fase per la parte potenziale (U_V)

        e la parte cinetica (U_T).
        """
      
        hbar = self.state.hbar
        m = self.state.m
        dt = self.dt
        # 1. Fase del Potenziale (Spazio Reale x): exp(-i * V(x) * dt / (2 * hbar))
        # Nota: usiamo dt/2 per lo step diviso (Half-step)
        self.U_V = np.exp(-1j * self.state.V * (dt / 2.0) / hbar)

        # 2. Fase Cinetica (Spazio dei Momenti p): exp(-i * p^2 * dt / (2 * m * hbar))

        self.U_T = np.exp(-1j * (self.state.p ** 2) * dt / (2.0 * m * hbar))

    def set_dt(self, new_dt: float):
        """updates time step and recomputes the operators"""
        self.dt = new_dt
        self._update_operators()

    def step(self):
        """execute a time step dt for the wave function"""
        psi = self.state.psi

        # 1. Primo mezzo passo nel potenziale (Spazio Reale)
        psi = psi * self.U_V

        # 2. Passo completo nell'energia cinetica (Spazio dei Momenti)
        # Passiamo allo spazio p con la FFT

        psi_p = np.fft.fft(psi)

        psi_p = psi_p * self.U_T

        # Torniamo allo spazio x con la IFFT (Trasformata Inversa)
        psi = np.fft.ifft(psi_p)

        psi = psi * self.U_V
        self.state.psi = psi
    


    

        