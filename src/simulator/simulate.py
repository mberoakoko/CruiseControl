import typing

import numpy as np

from src.simulator import trajectory
import control

def plant_simulation(plant: control.NonlinearIOSystem, dt: float, t_final: float) -> control.TimeResponseData:
    sim_trajectory = trajectory.create_test_trajectory(dt, t_final)
    return control.input_output_response(plant,sim_trajectory.t, np.r_[sim_trajectory.u, sim_trajectory.theta])

