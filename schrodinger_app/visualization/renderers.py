import matplotlib.pyplot as plt
import numpy as np
from core.state import QuantumState

class WaveFunctionRender:
    """Gestisce la rappresentazione grafica della funzione d'onda

    e del potenziale usando Matplotlib.
    """
    def __init__(self, state: QuantumState):
        self.state = state

        # Attiva la modalità interattiva di Matplotlib
        plt.ion()

        # Crea la figura e l'asse del grafico
        self.fig, self.ax = plt.subplots(figsize=(8,5))

        # Disegna le linee iniziali e salva i riferimenti per aggiornarle rapidamente
        (self.line_psi,) = self.ax.plot(
            state.x,
            state.probability_density,
            label=r"$|\Psi(x)|^2$",
            color="#1f77b4",
            linewidth=2,
        )

        # Scaliamo visivamente il potenziale per non farlo uscire dal grafico

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

        # Configurazione degli assi e dei titoli
        self.ax.set_xlim(-40, 40)
        self.ax.set_ylim(-0.02, 0.35)
        self.ax.set_xlabel("Posizione x")
        self.ax.set_ylabel(r"Densità di probabilità $|\Psi|^2$")
        self.ax.set_title("Simulazione dell'Equazione di Schrödinger 1D")
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right")

        self.fig.tight_layout()

    def update(self, current_step: int = 0):
        """Aggiorna solo i dati dei vettori y delle linee (molto più rapido di plt.clf())."""
        self.line_psi.set_ydata(self.state.probability_density)

        # Aggiorna il titolo con lo step corrente
        self.ax.set_title(
            f"Simulazione Equazione di Schrödinger 1D — Step: {current_step}"
        )

        # Ridisegna la figura e fa una piccolissima pausa per l'event loop della GUI

        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)  # Pausa di 1 ms per aggiornare la GUI 

    def close(self):
        """Chiude la finestra grafica."""
        plt.ioff()
        plt.close(self.fig)
