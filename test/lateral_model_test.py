import unittest

import numpy as np

from src.models.lateral_model import LateralVehicleModel, create_lateral_model

VEHICLE_MODEL_OBJECT = LateralVehicleModel()
PARAM_SET_VEHICLE_MODEL= (1, np.array([0]), np.array([0, 0]), None)
NON_LINEAR_IO_VEHICLE = create_lateral_model(VEHICLE_MODEL_OBJECT)

class LateralModelTest(unittest.TestCase):
    def test_vehicle_model_update(self):
        self.assertIsNotNone(VEHICLE_MODEL_OBJECT.update(*PARAM_SET_VEHICLE_MODEL))


class ControlNonLinearIOSystemTest(unittest.TestCase):
    def test_non_linear_io_update(self):
        self.assertIsNotNone(NON_LINEAR_IO_VEHICLE.dynamics(*PARAM_SET_VEHICLE_MODEL))
        self.assertIsNotNone(NON_LINEAR_IO_VEHICLE.output(*PARAM_SET_VEHICLE_MODEL))
        self.assertEqual(NON_LINEAR_IO_VEHICLE.nstates, 1)


if __name__ == '__main__':
    unittest.main()
