"""
Gradient-based (and baseline derivative-free) fitting of the drift matrix A on
the non-interventional dataset, split into independent components so each
design choice can be ablated on its own:

    problem.py      data container (snapshots, mu, sigma, dt, A_true)
    transition.py   one-step pushforward: exact OU (Van Loan) | Euler-Maruyama
    losses.py       sliced-W on particles | Gaussian-moment KL | Bures-W2
    params.py       how A is parameterized: full | positive diagonal
    optimizers.py   adam | lbfgs | cma | nelder_mead   (all share one signature)
    pipeline.py     Stage / Config: chain stages (e.g. moment fit -> SW refine),
                    multi-start, L1 penalty
    metrics.py      recovery metrics vs A_true + the "check 0" loss gap

Every component is picked by a string in a Stage/Config, so an ablation is a
list of Configs differing in exactly one field -- see run_gradfit_ablation.py.

float64 is enabled here, before anything touches jax.numpy: expm of the Van
Loan block matrix and the Cholesky of Q lose too much in float32.
"""
import jax

jax.config.update("jax_enable_x64", True)
