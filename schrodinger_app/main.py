import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.potentials import rectangular_barrier
from core.state import QuantumState
from physics.initial_conditions import gaussian_wavepacket
from solvers.split_operator import SplitStepSolver
from visualization.renderers import WavefunctionRenderer


def main():
    # 1. Inizializza lo stato quantistico
    state = QuantumState(N=512, L=100.0, hbar=1.0, m=1.0)

    # 2. Condizione iniziale e potenziale
    psi0 = gaussian_wavepacket(state.x, x0=-20.0, k0=3.0, sigma=3.0)
    state.set_wavefunction(psi0)

    V = rectangular_barrier(state.x, V0=5.0, width=4.0, center=0.0)
    state.set_potential(V)

    # 3. Solver e Renderer
    solver = SplitStepSolver(state, dt=0.05)
    renderer = WavefunctionRenderer(state)

    # 4. Usa FuncAnimation per gestire l'event loop di macOS in modo nativo
    anim = FuncAnimation(
        renderer.fig,
        renderer.animate_frame,
        fargs=(solver,),
        interval=20,  # 20 ms tra un frame e l'altro (~50 fps)
        blit=True,
    )

    print("Avvio della finestra grafica...")
    plt.show()


if __name__ == "__main__":
    main()