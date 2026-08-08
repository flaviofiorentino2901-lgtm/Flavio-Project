import numpy as np

class QuantumState:
    """Mantiene lo stato della griglia spaziale, degli impulsi,

    del potenziale V(x) e della funzione d'onda psi(x)."""
    def __init__(self,
                  N: int = 512,
                  L: float = 100.0,
                  hbar: float = 1.0,
                  m: float = 1.0
    ):
        self.N = N  # Numero di punti sulla griglia
        self.L = L  # Estensione del dominio spaziale (-L/2, L/2)
        self.hbar = hbar
        self.m = m

        # griglia spaziale

        self.dx = L/N
        self.x = np.linspace(- L / 2, L / 2, N, endpoint=False)

        # griglia degli impulsi tramite trasformata di fourier
        # k = 2*pi *frequenze_spaziali
        k = 2 * np.pi * np.fft.fftfreq(N, d = self.dx)
        self.p = hbar * k # impulso p = hbar * k

        # 3. Potenziale V(x) e Funzione d'onda psi(x) inizializzati a zero

        self.V = np.zeros(N, dtype = np.float64)
        self.psi = np.zeros(N, dtype = np.complex128)

    def set_potential(self, V_array: np.ndarray):
        if len(V_array) != self.N:
            raise ValueError(f"la funzione d'onda deve avere dimensione N = {self.N}")
        self.V = V_array.copy()

    def set_wavefunction(self, psi_array: np.ndarray):
        if len(psi_array) != self.N:
            raise ValueError(
                f"La funzione d'onda deve avere dimensione N={self.N}"
            )
        # Normalizzazione: \int |psi|^2 dx = 1
        norm = np.sqrt(np.sum(np.abs(psi_array) ** 2) * self.dx)
        if norm > 0:
            self.psi = psi_array / norm
        else:
            self.psi = psi_array.copy()  # Se la norma è zero, non normalizzare  

    @property
    def probability_density(self) -> np.ndarray:
        """Restituisce |psi(x)|^2 (densità di probabilità)."""
        return np.abs(self.psi) ** 2  