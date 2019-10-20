"""Test scaling data"""

import logging
import unittest
import numpy as np

from nptdms import TdmsFile, TdmsObject
from nptdms.log import log_manager


class TestTdmsFile(TdmsFile):
    def __init__(self):
        self.segments = []
        self.objects = {}
        self.memmap_dir = None


class ScalingDataTests(unittest.TestCase):
    def setUp(self):
        log_manager.set_level(logging.DEBUG)

    def test_no_scaling(self):
        """Test that raw data is returned unscaled when there is no scaling"""

        tdms_obj = TdmsObject("/'group'/'channel'")
        expected_data = np.array([1.0, 2.0, 3.0])
        tdms_obj._data = expected_data

        self.assertIs(expected_data, tdms_obj.data)

    def test_unsupported_scaling_type(self):
        """Raw data is returned unscaled when the scaling type is unsupported.
        """

        tdms_obj = TdmsObject("/'group'/'channel'")
        expected_data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "UnknownScaling"
        tdms_obj._data = expected_data

        self.assertIs(expected_data, tdms_obj.data)

    def test_linear_scaling(self):
        """Test linear scaling"""

        tdms_obj = TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[0]_Linear_Slope"] = 2.0
        tdms_obj.properties["NI_Scale[0]_Linear_Y_Intercept"] = 10.0

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_polynomial_scaling(self):
        """Test polynomial scaling"""

        tdms_obj = TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Polynomial"
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[0]"] = 10.0
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[1]"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[2]"] = 2.0
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[3]"] = 3.0

        expected_data = np.array([16.0, 44.0, 112.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_polynomial_scaling_with_3_coefficients(self):
        """Test polynomial scaling"""

        tdms_obj = TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Polynomial"
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients_Size"] = 3
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[0]"] = 10.0
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[1]"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Polynomial_Coefficients[2]"] = 2.0

        expected_data = np.array([13.0, 20.0, 31.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_multiple_scalings_applied_in_order(self):
        """Test all scalings applied from multiple scalings
        """

        tdms_obj = TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 3
        tdms_obj.properties["NI_Scaling_Status"] = "unscaled"
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[0]_Linear_Slope"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Linear_Y_Intercept"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Linear_Input_Source"] = 0xFFFFFFFF
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[1]_Linear_Slope"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Linear_Y_Intercept"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Linear_Input_Source"] = 0
        tdms_obj.properties["NI_Scale[2]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[2]_Linear_Slope"] = 3.0
        tdms_obj.properties["NI_Scale[2]_Linear_Y_Intercept"] = 3.0
        tdms_obj.properties["NI_Scale[2]_Linear_Input_Source"] = 1

        expected_data = np.array([21.0, 27.0, 33.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_multiple_scalings_but_all_with_raw_data_input(self):
        """Test that only the last scaling is applied from multiple scalings
           when it has the raw data as the input source
        """

        tdms_obj = TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 3
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[0]_Linear_Slope"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Linear_Y_Intercept"] = 1.0
        tdms_obj.properties["NI_Scale[0]_Linear_Input_Source"] = 0xFFFFFFFF
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[1]_Linear_Slope"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Linear_Y_Intercept"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Linear_Input_Source"] = 0xFFFFFFFF
        tdms_obj.properties["NI_Scale[2]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[2]_Linear_Slope"] = 3.0
        tdms_obj.properties["NI_Scale[2]_Linear_Y_Intercept"] = 3.0
        tdms_obj.properties["NI_Scale[2]_Linear_Input_Source"] = 0xFFFFFFFF

        expected_data = np.array([6.0, 9.0, 12.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_scaling_from_group(self):
        """Test linear scaling in a group"""

        tdms_file = TestTdmsFile()

        tdms_channel = TdmsObject("/'group'/'channel'", tdms_file)
        tdms_channel._data = np.array([1.0, 2.0, 3.0])

        tdms_group = TdmsObject("/'group'", tdms_file)
        tdms_group.properties["NI_Number_Of_Scales"] = 1
        tdms_group.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_group.properties["NI_Scale[0]_Linear_Slope"] = 2.0
        tdms_group.properties["NI_Scale[0]_Linear_Y_Intercept"] = 10.0

        tdms_file.objects[tdms_channel.path] = tdms_channel
        tdms_file.objects[tdms_group.path] = tdms_group

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_channel.data)

    def test_scaling_from_root(self):
        """Test linear scaling in the root object"""

        tdms_file = TestTdmsFile()

        tdms_channel = TdmsObject("/'group'/'channel'", tdms_file)
        tdms_channel._data = np.array([1.0, 2.0, 3.0])

        tdms_root = TdmsObject("/", tdms_file)
        tdms_root.properties["NI_Number_Of_Scales"] = 1
        tdms_root.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_root.properties["NI_Scale[0]_Linear_Slope"] = 2.0
        tdms_root.properties["NI_Scale[0]_Linear_Y_Intercept"] = 10.0

        tdms_file.objects[tdms_channel.path] = tdms_channel
        tdms_file.objects[tdms_root.path] = tdms_root

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_channel.data)


if __name__ == '__main__':
    unittest.main()
