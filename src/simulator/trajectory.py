import numpy as np
from numpy.typing import NDArray
import control
import dataclasses

@dataclasses.dataclass(frozen=True)
class LateralModelTrajectory:
    t: NDArray
    u: NDArray
    theta: NDArray


def create_test_trajectory(dt: float, t_final: float) -> LateralModelTrajectory:
    t: NDArray = np.arange(0, t_final, round(dt/t_final))
    theta: NDArray = np.zeros_like(t)
    theta[t<=6 & t >= 5] = 4./180. * np.pi * (t-5)
    return LateralModelTrajectory(
        t = t,
        u = 10 * np.ones_like(t),
        theta=theta
    )
