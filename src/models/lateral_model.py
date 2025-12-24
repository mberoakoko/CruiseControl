import control
import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class EngineModel:
    t_m: float = 190          #   engine torque constant
    omega_m: float = 120     #   peak angular speed
    beta: float = 0.4         #   peak angular rolloff


@dataclasses.dataclass
class LateralVehicleModel:
    m: float = 1000
    g: float = 9.81
    c_r: float = 0.1
    rho: float = 0.1
    c_d: float = 0.1
    area: float = 100

    alpha: list[float] = dataclasses.field(default_factory=lambda : [40, 25, 16, 12, 10])
    engine_model: EngineModel = dataclasses.field(default_factory=lambda: EngineModel())

    def motor_torque(self, omega: float | np.ndarray) -> float:
        Tm = self.engine_model.t_m
        omega_m = self.engine_model.omega_m
        beta  = self.engine_model.beta
        return np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2))

    def select_gear(self, velocity: float) -> int:
        omega_v = np.array(self.alpha) * velocity
        torques = self.motor_torque(omega_v)
        return int(np.argmax(torques) - 1)


    def update(self, t: float, x: np.ndarray, u: np.ndarray, params = None):
        v = x[0]
        throttle = np.clip(u[0], 0, 1)
        theta = u[1]
        gear = self.select_gear(v)
        omega = self.alpha[int(gear) - 1] * v
        F = self.alpha[int(gear) - 1] * self.motor_torque(omega) * throttle

        Fg = self.m * self.g * np.sin(theta)
        Fr = self.m * self.g * self.c_r * np.sign(v)
        Fa = 1 / 2 * self.rho * self.c_d * self.area * abs(v) * v
        Fd = Fg + Fr + Fa
        dv = (F - Fd)/self.m
        return dv


def create_lateral_model(vehicle_model: LateralVehicleModel) -> control.NonlinearIOSystem:
    return control.NonlinearIOSystem(
        vehicle_model.update, vehicle_model.output,
        inputs=('u', 'theta'), outputs=("v",),
        name="LateralVehicleModel",
        states=1
    )