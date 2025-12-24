import unittest
from src.models.lateral_model import LateralVehicleModel

VEHICLE_MODEL_OBJECT = LateralVehicleModel()
class LateralModelTest(unittest.TestCase):
    def test_vehicle_model_update(self):
        ...



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
