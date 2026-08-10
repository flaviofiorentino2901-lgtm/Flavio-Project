from abc import ABC, abstractmethod
from core.state import QuantumState


class BaseSolver(ABC):
    """Classe base astratta per gli integratori dell'equazione di Schrödinger."""

    def __init__(self, state: QuantumState, dt: float = 0.05):
        self.state = state
        self.dt = dt

    @abstractmethod
    def step(self):
        """Esegue un singolo avanzamento temporale dt sullo stato."""
        pass