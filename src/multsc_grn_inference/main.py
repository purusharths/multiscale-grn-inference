"""
Algorithm 1, function Main (paper lines 59-69):

    function Main()
        chi_i <- Preprocessing(D)
        theta0 <- BIPInitilize(D)
        # ...
        while Optimize do
            ComputeLoss
        # BIP_PosteriorEstimate(theta_hat)   # @todo
        return null

Note the paper's own pseudocode marks the final posterior-estimate step
`# @todo` and Main() literally `return null` -- even the algorithm spec
doesn't define what Main() ultimately hands back. This module wires
Preprocessing -> BIPInitialize -> Optimize; it does not implement the
undefined posterior-estimate step, and (unlike the pseudocode) returns
theta_hat rather than null, since a caller needs *something* usable.
"""
from __future__ import annotations

import numpy as np

from multsc_grn_inference.bip_initialize import bip_initialize
from multsc_grn_inference.optimize import optimize
from multsc_grn_inference.preprocessing import preprocessing
from multsc_grn_inference.theta import Theta


def main(D: list[np.ndarray], mu: list[np.ndarray], dt: float) -> Theta:
    """Wires Preprocessing -> BIPInitialize -> Optimize. See module docstring."""
    raise NotImplementedError
