"""Test scaling data"""

import logging
import numpy as np
import unittest

from nptdms import tdms


class TestTdmsFile(tdms.TdmsFile):
    def __init__(self):
        self.segments = []
        self.objects = {}
        self.memmap_dir = None


class ScalingDataTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

    def test_no_scaling(self):
        """Test that raw data is returned unscaled when there is no scaling"""

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        expected_data = np.array([1.0, 2.0, 3.0])
        tdms_obj._data = expected_data

        self.assertIs(expected_data, tdms_obj.data)

    def test_unsupported_scaling_type(self):
        """Raw data is returned unscaled when the scaling type is unsupported.
        """

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        expected_data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "UnknownScaling"
        tdms_obj._data = expected_data

        self.assertIs(expected_data, tdms_obj.data)

    def test_linear_scaling(self):
        """Test linear scaling"""

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[1]_Linear_Slope"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Linear_Y_Intercept"] = 10.0

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_polynomial_scaling(self):
        """Test polynomial scaling"""

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Polynomial"
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[0]"] = 10.0
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[1]"] = 1.0
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[2]"] = 2.0
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[3]"] = 3.0

        expected_data = np.array([16.0, 44.0, 112.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_polynomial_scaling_with_3_coefficients(self):
        """Test polynomial scaling"""

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 1
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Polynomial"
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients_Size"] = 3
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[0]"] = 10.0
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[1]"] = 1.0
        tdms_obj.properties["NI_Scale[1]_Polynomial_Coefficients[2]"] = 2.0

        expected_data = np.array([13.0, 20.0, 31.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_multiple_scalings(self):
        """Test correct scaling selected from multiple scalings"""

        tdms_obj = tdms.TdmsObject("/'group'/'channel'")
        tdms_obj._data = np.array([1.0, 2.0, 3.0])
        tdms_obj.properties["NI_Number_Of_Scales"] = 3
        tdms_obj.properties["NI_Scale[0]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[0]_Linear_Slope"] = 2.0
        tdms_obj.properties["NI_Scale[0]_Linear_Y_Intercept"] = 10.0
        tdms_obj.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[1]_Linear_Slope"] = 3.0
        tdms_obj.properties["NI_Scale[1]_Linear_Y_Intercept"] = 12.0
        tdms_obj.properties["NI_Scale[2]_Scale_Type"] = "Linear"
        tdms_obj.properties["NI_Scale[2]_Linear_Slope"] = 4.0
        tdms_obj.properties["NI_Scale[2]_Linear_Y_Intercept"] = 2.0

        expected_data = np.array([6.0, 10.0, 14.0])

        np.testing.assert_almost_equal(expected_data, tdms_obj.data)

    def test_scaling_from_group(self):
        """Test linear scaling in a group"""

        tdms_file = TestTdmsFile()

        tdms_channel = tdms.TdmsObject("/'group'/'channel'", tdms_file)
        tdms_channel._data = np.array([1.0, 2.0, 3.0])

        tdms_group = tdms.TdmsObject("/'group'", tdms_file)
        tdms_group.properties["NI_Number_Of_Scales"] = 1
        tdms_group.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_group.properties["NI_Scale[1]_Linear_Slope"] = 2.0
        tdms_group.properties["NI_Scale[1]_Linear_Y_Intercept"] = 10.0

        tdms_file.objects[tdms_channel.path] = tdms_channel
        tdms_file.objects[tdms_group.path] = tdms_group

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_channel.data)

    def test_scaling_from_root(self):
        """Test linear scaling in the root object"""

        tdms_file = TestTdmsFile()

        tdms_channel = tdms.TdmsObject("/'group'/'channel'", tdms_file)
        tdms_channel._data = np.array([1.0, 2.0, 3.0])

        tdms_root = tdms.TdmsObject("/", tdms_file)
        tdms_root.properties["NI_Number_Of_Scales"] = 1
        tdms_root.properties["NI_Scale[1]_Scale_Type"] = "Linear"
        tdms_root.properties["NI_Scale[1]_Linear_Slope"] = 2.0
        tdms_root.properties["NI_Scale[1]_Linear_Y_Intercept"] = 10.0

        tdms_file.objects[tdms_channel.path] = tdms_channel
        tdms_file.objects[tdms_root.path] = tdms_root

        expected_data = np.array([12.0, 14.0, 16.0])

        np.testing.assert_almost_equal(expected_data, tdms_channel.data)


if __name__ == '__main__':
    unittest.main()
