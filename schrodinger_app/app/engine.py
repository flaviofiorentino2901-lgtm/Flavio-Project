from core.state import QuantumState
from physics.observables import expected_position, norm
from solvers.base_solver import BaseSolver
from visualization.renderers import WavefunctionRenderer


class SimulationEngine:
    """Gestisce il loop principale di esecuzione della simulazione."""

    def __init__(
        self,
        state: QuantumState,
        solver: BaseSolver,
        renderer: WavefunctionRenderer,
    ):
        self.state = state
        self.solver = solver
        self.renderer = renderer
        self.is_running = False

    def run(self, steps: int = 1000, render_every: int = 2):
        """Esegue il ciclo temporale per il numero di step specificato."""
        self.is_running = True
        print(f"Avvio simulazione per {steps} step temporali...")

        try:
            for step in range(steps):
                if not self.is_running:
                    break

                # Avanza la fisica di uno step
                self.solver.step()

                # Aggiorna la grafica a intervalli regolari
                if step % render_every == 0:
                    self.renderer.update(current_step=step)

                    # Stampa a video alcuni osservabili ogni 50 step per monitorare la simulazione
                    if step % 50 == 0:
                        n = norm(self.state)
                        x_exp = expected_position(self.state)
                        print(
                            f"Step {step:4d} | Norma: {n:.6f} | <x>: {x_exp:+.3f}"
                        )

        except KeyboardInterrupt:
            print("\nSimulazione interrotta dall'utente.")
        finally:
            self.is_running = False
            self.renderer.close()
            print("Simulazione completata.")