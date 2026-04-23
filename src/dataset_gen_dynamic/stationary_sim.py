import numpy as np

class NetworkSimulator:
    def lebann__init__(self, num_genes=4, network_density=0.3):
        self.num_genes = num_genes
        self.generate_network(network_density)
    
    def generate_network(self, density):
        print(f"Generating {self.num_genes}-gene network...")
        
        self.A = np.zeros((self.num_genes, self.num_genes))
        
        # Stronger self-regulation for better identifiability
        for i in range(self.num_genes):
            self.A[i, i] = np.random.uniform(1.0, 1.8)
            # self.A[i, i] = -np.random.normal(0.0, 1.)
        
        n_edges = int(density * self.num_genes * (self.num_genes - 1))
        print(n_edges)
        edges_added = 0
        
        while edges_added < n_edges:
            i, j = np.random.choice(self.num_genes, 2, replace=False)
            if self.A[i, j] == 0:
                sign = 1 if np.random.random() > 0.3 else -1  # More positive regulation
                strength = np.random.uniform(0.3, 1.2)  # Moderate strengths
                self.A[i, j] = sign * strength
                edges_added += 1
                
        # self.mu = np.random.normal(2.5, 1.0, self.num_genes)  # Tighter range
        # self.D = np.diag(np.random.normal(0.0, 0.2, self.num_genes))  # Less noise
        # should there be a relation between the values of mu, D and A?
        self.mu = np.random.uniform(2.5, 3.5, self.num_genes)  # Tighter range
        self.D = np.diag(np.random.uniform(0.1, 0.2, self.num_genes))  # Less noise
        
        print(f"Generated {np.sum(np.abs(self.A) > 0.1)} regulatory relationships")
        print(f"Enhanced ground truth A matrix:")
        print(self.A)
    
    def simulate(self, T=4.0, num_samples=500, save_every=15):
        """numerical integration"""
        print(f"Simulating {num_samples} trajectories...")
        
        dt = 0.005  
        steps = int(T / dt)
        save_steps = steps // save_every
        
        data = np.zeros((num_samples, save_steps + 1, self.num_genes))
        
        for sample in range(num_samples):
            # more varied initial conditions
            X = self.mu + 0.5 * np.random.randn(self.num_genes)
            X = np.maximum(X, 0.1)  # ensure positive start
            data[sample, 0] = X.copy()
            
            save_idx = 1
            for t in range(1, steps):
                # SDE integration
                drift = self.A @ (self.mu - X) # A(mu - X_t)
                noise = np.sqrt(np.diag(self.D) * dt) * np.random.randn(self.num_genes)
                X += drift * dt + noise
                
                #  positivity constraint
                X = np.maximum(X, 0.05)
                
                if t % save_every == 0 and save_idx < save_steps + 1:
                    data[sample, save_idx] = X.copy()
                    save_idx += 1
        
        print(f"Done: simulation. Data shape: {data.shape}")
        return data