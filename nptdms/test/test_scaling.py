"""Test scaling data"""

import numpy as np
import pytest

from nptdms.scaling import get_scaling

try:
    import thermocouples_reference
except ImportError:
    thermocouples_reference = None
try:
    import scipy
except ImportError:
    scipy = None


def test_unsupported_scaling_type():
    """Raw data is returned unscaled when the scaling type is unsupported.
    """

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "UnknownScaling"
    }
    scaling = get_scaling(properties, {}, {})

    assert scaling is None


def test_linear_scaling():
    """Test linear scaling"""

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_polynomial_scaling():
    """Test polynomial scaling"""

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([16.0, 44.0, 112.0])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Polynomial",
        "NI_Scale[0]_Polynomial_Coefficients[0]": 10.0,
        "NI_Scale[0]_Polynomial_Coefficients[1]": 1.0,
        "NI_Scale[0]_Polynomial_Coefficients[2]": 2.0,
        "NI_Scale[0]_Polynomial_Coefficients[3]": 3.0,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_polynomial_scaling_with_3_coefficients():
    """Test polynomial scaling"""

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([13.0, 20.0, 31.0])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Polynomial",
        "NI_Scale[0]_Polynomial_Coefficients_Size": 3,
        "NI_Scale[0]_Polynomial_Coefficients[0]": 10.0,
        "NI_Scale[0]_Polynomial_Coefficients[1]": 1.0,
        "NI_Scale[0]_Polynomial_Coefficients[2]": 2.0,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_rtd_scaling():
    """Test RTD scaling"""

    data = np.array([0.5, 0.6])
    expected_scaled_data = np.array([1256.89628, 1712.83429])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "RTD",
        "NI_Scale[0]_RTD_Current_Excitation": 0.001,
        "NI_Scale[0]_RTD_R0_Nominal_Resistance": 100.0,
        "NI_Scale[0]_RTD_A": 0.0039083,
        "NI_Scale[0]_RTD_B": -5.775e-07,
        "NI_Scale[0]_RTD_C": -4.183e-12,
        "NI_Scale[0]_RTD_Lead_Wire_Resistance": 0.0,
        "NI_Scale[0]_RTD_Resistance_Configuration": 3,
        "NI_Scale[0]_RTD_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(
        expected_scaled_data, scaled_data, decimal=3)


def test_table_scaling():
    """Test table scaling"""

    data = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5])
    expected_scaled_data = np.array([2.0, 2.0, 3.0, 6.0, 8.0, 8.0])

    # The scaled values are actually the range of inputs into the scaling,
    # which are mapped to the pre-scaled values. This makes no sense but
    # matches the behaviour of the Excel TDMS plugin.

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Table",
        "NI_Scale[0]_Table_Scaled_Values_Size": 3,
        "NI_Scale[0]_Table_Scaled_Values[0]": 1.0,
        "NI_Scale[0]_Table_Scaled_Values[1]": 2.0,
        "NI_Scale[0]_Table_Scaled_Values[2]": 3.0,
        "NI_Scale[0]_Table_Pre_Scaled_Values_Size": 3,
        "NI_Scale[0]_Table_Pre_Scaled_Values[0]": 2.0,
        "NI_Scale[0]_Table_Pre_Scaled_Values[1]": 4.0,
        "NI_Scale[0]_Table_Pre_Scaled_Values[2]": 8.0,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_add_scaling():
    """ Test scaling that adds two input scalings"""

    scaler_data = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([2.0, 4.0, 6.0]),
    }
    expected_scaled_data = np.array([3.0, 6.0, 9.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scale[2]_Scale_Type": "Add",
        "NI_Scale[2]_Add_Left_Operand_Input_Source": 0,
        "NI_Scale[2]_Add_Right_Operand_Input_Source": 1,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale_daqmx(scaler_data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_subtract_scaling():
    """ Test scaling that subtracts an input scaling from another"""

    # This behaves the opposite to what you'd expect, the left operand
    # is subtracted from the right operand.
    scaler_data = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([2.0, 4.0, 6.0]),
    }
    expected_scaled_data = np.array([1.0, 2.0, 3.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scale[2]_Scale_Type": "Subtract",
        "NI_Scale[2]_Subtract_Left_Operand_Input_Source": 0,
        "NI_Scale[2]_Subtract_Right_Operand_Input_Source": 1,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale_daqmx(scaler_data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


@pytest.mark.skipif(thermocouples_reference is None, reason="thermocouples_reference is not installed")
@pytest.mark.skipif(scipy is None, reason="scipy is not installed")
def test_thermocouple_scaling_voltage_to_temperature():
    """Test thermocouple scaling from a voltage in uV to temperature"""

    data = np.array([
        0.0, 10.0, 100.0, 1000.0])
    expected_scaled_data = np.array([
        0.0, 0.2534448,  2.5309141, 24.9940185])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermocouple",
        "NI_Scale[0]_Thermocouple_Thermocouple_Type": 10073,
        "NI_Scale[0]_Thermocouple_Scaling_Direction": 0,
        "NI_Scale[0]_Thermocouple_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(
        expected_scaled_data, scaled_data, decimal=3)


@pytest.mark.skipif(thermocouples_reference is None, reason="thermocouples_reference is not installed")
def test_thermocouple_scaling_temperature_to_voltage():
    """Test thermocouple scaling from a temperature to voltage in uV"""

    data = np.array([
        0.0, 10.0, 50.0, 100.0])
    expected_scaled_data = np.array([
        0.0, 396.8619078, 2023.0778862, 4096.2302187])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermocouple",
        "NI_Scale[0]_Thermocouple_Thermocouple_Type": 10073,
        "NI_Scale[0]_Thermocouple_Scaling_Direction": 1,
        "NI_Scale[0]_Thermocouple_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(
        expected_scaled_data, scaled_data, decimal=3)


def test_multiple_scalings_applied_in_order():
    """Test all scalings applied from multiple scalings
    """

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([21.0, 27.0, 33.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scaling_Status": "unscaled",
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 1.0,
        "NI_Scale[0]_Linear_Y_Intercept": 1.0,
        "NI_Scale[0]_Linear_Input_Source": 0xFFFFFFFF,
        "NI_Scale[1]_Scale_Type": "Linear",
        "NI_Scale[1]_Linear_Slope": 2.0,
        "NI_Scale[1]_Linear_Y_Intercept": 2.0,
        "NI_Scale[1]_Linear_Input_Source": 0,
        "NI_Scale[2]_Scale_Type": "Linear",
        "NI_Scale[2]_Linear_Slope": 3.0,
        "NI_Scale[2]_Linear_Y_Intercept": 3.0,
        "NI_Scale[2]_Linear_Input_Source": 1,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_multiple_scalings_but_all_with_raw_data_input():
    """Test that only the last scaling is applied from multiple scalings
       when it has the raw data as the input source
    """

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([6.0, 9.0, 12.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 1.0,
        "NI_Scale[0]_Linear_Y_Intercept": 1.0,
        "NI_Scale[0]_Linear_Input_Source": 0xFFFFFFFF,
        "NI_Scale[1]_Scale_Type": "Linear",
        "NI_Scale[1]_Linear_Slope": 2.0,
        "NI_Scale[1]_Linear_Y_Intercept": 2.0,
        "NI_Scale[1]_Linear_Input_Source": 0xFFFFFFFF,
        "NI_Scale[2]_Scale_Type": "Linear",
        "NI_Scale[2]_Linear_Slope": 3.0,
        "NI_Scale[2]_Linear_Y_Intercept": 3.0,
        "NI_Scale[2]_Linear_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_scaling_from_group():
    """Test linear scaling in a group"""

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    group_properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0,
    }
    scaling = get_scaling({}, group_properties, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)


def test_scaling_from_root():
    """Test linear scaling in the root object"""

    data = np.array([1.0, 2.0, 3.0])
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    root_properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0,
    }
    scaling = get_scaling({}, {}, root_properties)
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(expected_scaled_data, scaled_data)
