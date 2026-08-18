import numpy as np


# constant mu(t) = mu0
def mu_constant(mu0: np.ndarray, **kwargs) -> np.ndarray:
    return mu0.copy()

# linear mu(t) = mu0 + (amp * t/T) * direction
def mu_linear(
    mu0: np.ndarray,
    t: float,
    T: float,
    drift_dir: np.ndarray,
    drift_amp: float,
    **kwargs,
) -> np.ndarray:
    s = t / max(T, 1e-12)
    return mu0 + (drift_amp * s) * drift_dir


# delta: np.ndarray (num_genes,) — shift applied at t_star
# t_star: float — intervention time (default T/2 if not given)
# k: float — sharpness for sigmoid (default 20.0)

# sigmoid     mu(t) = mu0 + delta * σ(k*(t - t_star))
def mu_sigmoid(
    mu0: np.ndarray,
    t: float,
    T: float,
    delta: np.ndarray | None = None,
    t_star: float | None = None,
    k: float = 20.0,
    **kwargs,
) -> np.ndarray:
    if delta is None:
        delta = np.ones_like(mu0)
    if t_star is None:
        t_star = T / 2.0
    h = 1.0 / (1.0 + np.exp(-k * (t - t_star)))
    return mu0 + delta * h




# inverse sigmoid (knockout)   mu(t) = mu0 - delta * σ(k*(t - t_star))
# Mirror image of mu_sigmoid: models an intervention that *suppresses* the
# expression target after t_star rather than raising it.  Unlike mu_sigmoid,
# delta defaults to mu0 itself, so the default is a full knockout driving the
# target mu0 -> ~0 (a partial knockdown is delta < mu0).
def mu_inverse_sigmoid(
    mu0: np.ndarray,
    t: float,
    T: float,
    delta: np.ndarray | None = None,
    t_star: float | None = None,
    k: float = 20.0,
    **kwargs,
) -> np.ndarray:
    if delta is None:
        delta = mu0.copy()
    if t_star is None:
        t_star = T / 2.0
    h = 1.0 / (1.0 + np.exp(-k * (t - t_star)))
    return mu0 - delta * h


# heaviside mu(t) = mu0 + delta * H(t - t_star)
# piecewise mu(t) = mu0 + delta * (t >= t_star)
def mu_heaviside(
    mu0: np.ndarray,
    t: float,
    T: float,
    delta: np.ndarray | None = None,
    t_star: float | None = None,
    **kwargs,
) -> np.ndarray:
    if delta is None:
        delta = np.ones_like(mu0)
    if t_star is None:
        t_star = T / 2.0
    h = 1.0 if t >= t_star else 0.0
    return mu0 + delta * h
