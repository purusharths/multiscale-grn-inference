"""Entry point: python -m dataset_gen_dynamic"""

from dataset_gen_dynamic.simulator import OUParams, simulate
from dataset_gen_dynamic.visualizer import plot_dynamics, plot_embeddings


def main() -> None:
    print("Simulating mRNA expression via OU process …")
    expression, metadata, trajectory = simulate(
        n_cells=1000,
        n_genes=50,
        n_states=3,
        n_timepoints=200,
        ou_params=OUParams(theta=2.0, sigma=0.5, dt=0.05),
    )
    n_cells, n_genes = expression.shape
    print(f"  {n_cells} cells × {n_genes} genes")
    print(f"  State counts: {metadata['state'].value_counts().to_dict()}")

    print("Plotting OU dynamics …")
    plot_dynamics(trajectory, n_genes=10, output="dynamics.png")

    print("Plotting UMAP + PHATE embeddings …")
    plot_embeddings(expression, metadata, output="embeddings.png")

    print("Done.")


if __name__ == "__main__":
    main()
