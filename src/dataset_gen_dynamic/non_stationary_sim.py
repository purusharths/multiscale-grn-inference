import numpy as np

from .housekeeping.enforce_diagonal_dominance import enforce_diagonal_dominance
from .housekeeping.mu_options import mu_constant, mu_heaviside, mu_linear, mu_sigmoid


class NetworkSimulatorNonStationaryMu:
    def __init__(
        self,
        num_genes=10,
        network_density=0.3,
        seed=0,
        mu_mode="sigmoid",
        mu_kwargs=None,#for sigmoid, heaviside
    ):
        # init seed 
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.num_genes = num_genes
        self.mu_mode = mu_mode
        self.mu_kwargs = mu_kwargs or {}
        
        self.A = np.zeros((num_genes, num_genes))
        # atm, non zero diagonal + zero off-diagonal (more structure here? - TFs, modules, etc.))
        for i in range(num_genes):
            self.A[i, i] = rng.uniform(1.0, 1.8)

        n_edges = int(network_density * num_genes * (num_genes - 1)) #total edges as a fn of networkdenisty and genes. 
        edges_added = 0
        while edges_added < n_edges:
            i, j = rng.choice(num_genes, 2, replace=False) 
            if self.A[i, j] == 0:
                sign = 1 if rng.random() > 0.3 else -1 # 70% activation prob. 30% inibihiton
                strength = rng.uniform(0.3, 1.2) #change strength prob?? 
                self.A[i, j] = sign * strength
                edges_added += 1

        enforce_diagonal_dominance(self.A) #gershgorin cirlce thm

        self.mu0 = rng.uniform(2.5, 3.5, num_genes)
        self.D = np.diag(rng.uniform(0.1, 0.2, num_genes))

        # linear-drift parameters(mu_mode == "linear")
        direction = rng.normal(size=num_genes)
        direction /= (np.linalg.norm(direction) + 1e-12)
        self.mu_drift_dir = direction
        self.mu_drift_amp = 4.0


    def mu_t(self, t, T):
        mode = self.mu_mode
        kw = self.mu_kwargs

        if mode == "constant":
            return mu_constant(self.mu0)

        elif mode == "linear":
            return mu_linear(self.mu0, t, T, self.mu_drift_dir, self.mu_drift_amp)

        elif mode == "sigmoid":
            return mu_sigmoid(self.mu0, t, T, **kw)

        elif mode in ("heaviside", "piecewise"):
            return mu_heaviside(self.mu0, t, T, **kw)

        else:
            raise ValueError(
                f"Unknown mu_mode '{mode}'. "
                "Choose from: 'constant', 'linear', 'sigmoid', 'heaviside', 'piecewise'."
            )
        
    
    def simulate(self, T=4.0, num_samples=500, save_every=15, dt=0.005):
        steps = int(T / dt)
        save_steps = steps // save_every

        data = np.zeros((num_samples, save_steps + 1, self.num_genes))
        time_grid = np.linspace(0.0, T, save_steps + 1)

        for s in range(num_samples):
            mu_init = self.mu_t(0.0, T)
            X = mu_init + 0.5 * self.rng.normal(size=self.num_genes)
            X = np.maximum(X, 0.1)
            data[s, 0] = X.copy()

            save_idx = 1
            for step in range(1, steps):
                t = step * dt
                mu = self.mu_t(t, T)

                drift = self.A @ (mu - X)
                noise = np.sqrt(np.diag(self.D) * dt) * self.rng.normal(size=self.num_genes)
                X += drift * dt + noise
                X = np.maximum(X, 0.05)

                if step % save_every == 0 and save_idx < save_steps + 1:
                    data[s, save_idx] = X.copy()
                    save_idx += 1

        return data, time_grid
