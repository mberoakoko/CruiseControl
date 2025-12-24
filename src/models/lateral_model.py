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
    alpha: list[float] = dataclasses.field(default_factory=lambda : [40, 25, 16, 12, 10])
    engine_model: EngineModel = dataclasses.field(default_factory=lambda: EngineModel())

    def motor_torque(self, omega: float | np.ndarray) -> float:
        Tm = self.engine_model.t_m
        omega_m = self.engine_model.omega_m
        beta  = self.engine_model.beta
        return np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2))

    def select_gear(self, velocity: float, omega: float) -> int:
        omega_v = np.array(omega) * velocity
        torques = self.motor_torque(omega_v)
        return int(np.argmax(torques) - 1)


    def update(self, t: float, x: np.ndarray, u: np.ndarray, params = None):
        pass

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: None):
        pass


def create_lateral_model(vehicle_model: LateralVehicleModel) -> control.NonlinearIOSystem:
    return control.NonlinearIOSystem(
        vehicle_model.update, vehicle_model.output,
        inputs=[], outputs=[],
        name="LateralVehicleModel",
        states=5
    )