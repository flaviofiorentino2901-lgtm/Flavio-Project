import matplotlib.pyplot as plt
import numpy as np
from core.state import QuantumState


class WavefunctionRenderer:

    def __init__(self, state: QuantumState):
        self.state = state

        # Inizializza la figura
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        (self.line_psi,) = self.ax.plot(
            state.x,
            state.probability_density,
            label=r"$|\Psi(x)|^2$",
            color="#1f77b4",
            linewidth=2,
        )

        v_max = np.max(np.abs(state.V))
        v_scale = 0.2 / v_max if v_max > 0 else 1.0

        (self.line_v,) = self.ax.plot(
            state.x,
            state.V * v_scale,
            label="Potenziale V(x) (scalato)",
            color="#d62728",
            linestyle="--",
            alpha=0.7,
        )

        self.ax.set_xlim(-40, 40)
        self.ax.set_ylim(-0.02, 0.35)
        self.ax.set_xlabel("Posizione x")
        self.ax.set_ylabel(r"Densità di probabilità $|\Psi|^2$")
        self.ax.set_title("Simulazione dell'Equazione di Schrödinger 1D")
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()

    def animate_frame(self, frame, solver):
        """Funzione chiamata da FuncAnimation ad ogni frame."""
        # Avanza la simulazione di 2 step fisici per ogni frame grafico
        solver.step()
        solver.step()

        self.line_psi.set_ydata(self.state.probability_density)
        self.ax.set_title(
            f"Simulazione Equazione di Schrödinger 1D — Step: {frame * 2}"
        )
        return (self.line_psi,)