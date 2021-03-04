"""Test scaling data"""

import numpy as np
import pytest

from nptdms import types
from nptdms.scaling import get_scaling


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

    data = StubTdmsData(np.array([1, 2, 3], dtype=np.dtype('int32')))
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    properties = {
        "NI_Scaling_Status": "unscaled",
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    assert scaling.get_dtype(types.Int32, None) == np.dtype('float64')
    assert scaled_data.dtype == np.dtype('float64')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_polynomial_scaling():
    """Test polynomial scaling"""

    data = StubTdmsData(np.array([1, 2, 3], dtype=np.dtype('int32')))
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

    assert scaling.get_dtype(types.Int32, None) == np.dtype('float64')
    assert scaled_data.dtype == np.dtype('float64')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_polynomial_scaling_with_no_coefficients():
    """Test polynomial scaling when there are no coefficients, so data should be all zero
    """
    data = StubTdmsData(np.array([1.0, 2.0, 3.0]))
    expected_scaled_data = np.array([0.0, 0.0, 0.0])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Polynomial",
        "NI_Scale[0]_Polynomial_Coefficients_Size": 0
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_polynomial_scaling_with_3_coefficients():
    """Test polynomial scaling"""

    data = StubTdmsData(np.array([1, 2, 3], dtype=np.dtype('int32')))
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

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


@pytest.mark.parametrize(
    "resistance_configuration,lead_resistance,expected_data",
    [
        (2, 0.0, [1256.89628, 1712.83429]),
        (2, 100.0, [557.6879004146, 882.7374139697]),
        (3, 0.0, [1256.89628, 1712.83429]),
        (3, 100.0, [882.7374139697, 1256.896275222]),
        (4, 0.0, [1256.89628, 1712.83429]),
        (4, 100.0, [1256.89628, 1712.83429]),
    ]
)
def test_rtd_scaling(resistance_configuration, lead_resistance, expected_data):
    """Test RTD scaling"""

    data = StubTdmsData(np.array([0.5, 0.6]))

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "RTD",
        "NI_Scale[0]_RTD_Current_Excitation": 0.001,
        "NI_Scale[0]_RTD_R0_Nominal_Resistance": 100.0,
        "NI_Scale[0]_RTD_A": 0.0039083,
        "NI_Scale[0]_RTD_B": -5.775e-07,
        "NI_Scale[0]_RTD_C": -4.183e-12,
        "NI_Scale[0]_RTD_Lead_Wire_Resistance": lead_resistance,
        "NI_Scale[0]_RTD_Resistance_Configuration": resistance_configuration,
        "NI_Scale[0]_RTD_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    assert scaling.get_dtype(types.DoubleFloat, None) == np.dtype('float64')
    np.testing.assert_almost_equal(scaled_data, expected_data, decimal=3)


def test_rtd_scaling_with_negative_temperature():
    """ Test RTD scaling with negative temperature values, which requires
        solving the full quartic Callendar-Van Dusen equation
    """
    data = StubTdmsData(np.array([
        0.08, 0.09, 0.095, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]))
    expected_data = np.array([
        -50.77114, -25.48835, -12.76894, -0., 51.56605, 103.94273,
        157.1695, 211.28915, 266.34819, 322.3973, 379.49189])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "RTD",
        "NI_Scale[0]_RTD_Current_Excitation": 0.001,
        "NI_Scale[0]_RTD_R0_Nominal_Resistance": 100.0,
        "NI_Scale[0]_RTD_A": 0.0039083,
        "NI_Scale[0]_RTD_B": -5.775e-07,
        "NI_Scale[0]_RTD_C": -4.183e-12,
        "NI_Scale[0]_RTD_Lead_Wire_Resistance": 0,
        "NI_Scale[0]_RTD_Resistance_Configuration": 2,
        "NI_Scale[0]_RTD_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    assert scaling.get_dtype(types.DoubleFloat, None) == np.dtype('float64')
    assert scaled_data.dtype == np.dtype('float64')
    np.testing.assert_almost_equal(scaled_data, expected_data, decimal=5)


def test_table_scaling():
    """Test table scaling"""

    data = StubTdmsData(np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5]))
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

    assert scaling.get_dtype(types.DoubleFloat, None) == np.dtype('float64')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_add_scaling():
    """ Test scaling that adds two input scalings"""

    scaler_data = StubDaqmxData({
        0: np.array([1, 2, 3], dtype=np.dtype('int32')),
        1: np.array([2, 4, 6], dtype=np.dtype('uint32')),
    })
    expected_scaled_data = np.array([3.0, 6.0, 9.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scale[2]_Scale_Type": "Add",
        "NI_Scale[2]_Add_Left_Operand_Input_Source": 0,
        "NI_Scale[2]_Add_Right_Operand_Input_Source": 1,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(scaler_data)

    assert scaling.get_dtype(None, {0: types.Int32, 1: types.Uint32}) == np.dtype('int64')
    assert scaled_data.dtype == np.dtype('int64')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_add_scaling_with_default_inputs():
    """ Test scaling that adds two input scalings"""

    data = StubTdmsData(np.array([1, 2, 3], dtype=np.dtype('int32')))
    expected_scaled_data = np.array([2, 4, 6])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Add",
        "NI_Scale[0]_Add_Left_Operand_Input_Source": 0xFFFFFFFF,
        "NI_Scale[0]_Add_Right_Operand_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    assert scaling.get_dtype(types.Int32, None) == np.dtype('int32')
    assert scaled_data.dtype == np.dtype('int32')
    np.testing.assert_equal(expected_scaled_data, scaled_data)


def test_subtract_scaling():
    """ Test scaling that subtracts an input scaling from another"""

    # This behaves the opposite to what you'd expect, the left operand
    # is subtracted from the right operand.
    scaler_data = StubDaqmxData({
        0: np.array([1, 2, 3], dtype=np.dtype('int32')),
        1: np.array([2, 4, 6], dtype=np.dtype('uint32')),
    })
    expected_scaled_data = np.array([1.0, 2.0, 3.0])

    properties = {
        "NI_Number_Of_Scales": 3,
        "NI_Scale[2]_Scale_Type": "Subtract",
        "NI_Scale[2]_Subtract_Left_Operand_Input_Source": 0,
        "NI_Scale[2]_Subtract_Right_Operand_Input_Source": 1,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(scaler_data)

    assert scaling.get_dtype(None, {0: types.Int32, 1: types.Uint32}) == np.dtype('int64')
    assert scaled_data.dtype == np.dtype('int64')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_thermocouple_scaling_voltage_to_temperature():
    """Test thermocouple scaling from a voltage in uV to temperature"""

    data = StubTdmsData(np.array([0.0, 10.0, 100.0, 1000.0]))
    expected_scaled_data = np.array([0.000000, 0.250843, 2.508899, 24.983648])

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
        scaled_data, expected_scaled_data, decimal=3)


def test_thermocouple_scaling_temperature_to_voltage():
    """Test thermocouple scaling from a temperature to voltage in uV"""

    data = StubTdmsData(np.array([0.0, 10.0, 50.0, 100.0]))
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
        scaled_data, expected_scaled_data, decimal=3)


def test_thermocouple_scaling_voltage_to_temperature_benchmark(benchmark):
    """Test thermocouple scaling from a voltage in uV to temperature"""

    data = StubTdmsData(np.tile(np.array([0.0, 10.0, 100.0, 1000.0]), 100))
    expected_scaled_data = np.tile(np.array([0.000000, 0.250843, 2.508899, 24.983648]), 100)

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermocouple",
        "NI_Scale[0]_Thermocouple_Thermocouple_Type": 10073,
        "NI_Scale[0]_Thermocouple_Scaling_Direction": 0,
        "NI_Scale[0]_Thermocouple_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = benchmark(scaling.scale, data)

    np.testing.assert_almost_equal(
        scaled_data, expected_scaled_data, decimal=3)


def test_thermocouple_scaling_temperature_to_voltage_benchmark(benchmark):
    """Test thermocouple scaling from a temperature to voltage in uV"""

    data = StubTdmsData(np.tile(np.array([0.0, 10.0, 50.0, 100.0]), 100))
    expected_scaled_data = np.tile(np.array([
        0.0, 396.8619078, 2023.0778862, 4096.2302187]), 100)

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermocouple",
        "NI_Scale[0]_Thermocouple_Thermocouple_Type": 10073,
        "NI_Scale[0]_Thermocouple_Scaling_Direction": 1,
        "NI_Scale[0]_Thermocouple_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = benchmark(scaling.scale, data)

    np.testing.assert_almost_equal(
        scaled_data, expected_scaled_data, decimal=3)


@pytest.mark.parametrize(
    "resistance_configuration,lead_resistance,expected_data",
    [
        (2, 0.0, [287.1495569816, 290.71633623, 294.4862276706]),
        (2, 100.0, [287.1495569816, 290.71633623, 294.4862276706]),
        (3, 0.0, [287.1495569816, 290.71633623, 294.4862276706]),
        (3, 100.0, [287.4248927942, 291.0482875767, 294.8892119392]),
        (4, 0.0, [287.1495569816, 290.71633623, 294.4862276706]),
        (4, 100.0, [287.1495569816, 290.71633623, 294.4862276706]),
    ]
)
def test_thermistor_scaling_with_voltage_excitation(
        resistance_configuration, lead_resistance, expected_data):
    data = StubTdmsData(np.array([1.1, 1.0, 0.9]))

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermistor",
        "NI_Scale[0]_Thermistor_Resistance_Configuration": resistance_configuration,
        "NI_Scale[0]_Thermistor_Excitation_Type": 10322,
        "NI_Scale[0]_Thermistor_Excitation_Value": 2.5,
        "NI_Scale[0]_Thermistor_R1_Reference_Resistance": 10000.0,
        "NI_Scale[0]_Thermistor_Lead_Wire_Resistance": lead_resistance,
        "NI_Scale[0]_Thermistor_A": 0.0012873851,
        "NI_Scale[0]_Thermistor_B": 0.00023575235,
        "NI_Scale[0]_Thermistor_C": 9.497806e-8,
        "NI_Scale[0]_Thermistor_Temperature_Offset": 1.0,
        "NI_Scale[0]_Thermistor_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_data)


@pytest.mark.parametrize(
    "resistance_configuration,lead_resistance,expected_data",
    [
        (2, 0.0, [335.5876272527, 338.303823856, 341.3530400858]),
        (2, 100.0, [341.3530400858, 344.8212218133, 348.831282405]),
        (3, 0.0, [335.5876272527, 338.303823856, 341.3530400858]),
        (3, 100.0, [338.303823856, 341.3530400858, 344.8212218133]),
        (4, 0.0, [335.5876272527, 338.303823856, 341.3530400858]),
        (4, 100.0, [335.5876272527, 338.303823856, 341.3530400858]),
    ]
)
def test_thermistor_scaling_with_current_excitation(
        resistance_configuration, lead_resistance, expected_data):
    data = StubTdmsData(np.array([1.1, 1.0, 0.9]))

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermistor",
        "NI_Scale[0]_Thermistor_Resistance_Configuration": resistance_configuration,
        "NI_Scale[0]_Thermistor_Excitation_Type": 10134,
        "NI_Scale[0]_Thermistor_Excitation_Value": 1.0e-3,
        "NI_Scale[0]_Thermistor_R1_Reference_Resistance": 0.0,
        "NI_Scale[0]_Thermistor_Lead_Wire_Resistance": lead_resistance,
        "NI_Scale[0]_Thermistor_A": 0.0012873851,
        "NI_Scale[0]_Thermistor_B": 0.00023575235,
        "NI_Scale[0]_Thermistor_C": 9.497806e-8,
        "NI_Scale[0]_Thermistor_Temperature_Offset": 1.0,
        "NI_Scale[0]_Thermistor_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_data)


def test_thermistor_scaling_with_invalid_excitation_type():
    data = StubTdmsData(np.array([1.1, 1.0, 0.9]))

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Thermistor",
        "NI_Scale[0]_Thermistor_Resistance_Configuration": 3,
        "NI_Scale[0]_Thermistor_Excitation_Type": 12345,
        "NI_Scale[0]_Thermistor_Excitation_Value": 2.5,
        "NI_Scale[0]_Thermistor_R1_Reference_Resistance": 10000.0,
        "NI_Scale[0]_Thermistor_Lead_Wire_Resistance": 0.0,
        "NI_Scale[0]_Thermistor_A": 0.0012873851,
        "NI_Scale[0]_Thermistor_B": 0.00023575235,
        "NI_Scale[0]_Thermistor_C": 9.497806e-8,
        "NI_Scale[0]_Thermistor_Temperature_Offset": 1.0,
        "NI_Scale[0]_Thermistor_Input_Source": 0xFFFFFFFF,
    }
    scaling = get_scaling(properties, {}, {})
    with pytest.raises(ValueError) as exc_info:
        _ = scaling.scale(data)
    assert "Invalid excitation type: 12345" in str(exc_info.value)


def test_strain_scaling_full_bridge_type_i():
    _strain_scaling_test(
        [0.00068827, 0.00068036, 0.000688, 0.00068545, 0.00069104,
         0.00068033, 0.00068023, 0.00068316, 0.00067672, 0.000679],
        [-0.0001311, -0.00012959, -0.00013105, -0.00013056, -0.00013163,
         -0.00012959, -0.00012957, -0.00013013, -0.0001289, -0.00012933])


def test_strain_scaling_full_bridge_type_i_with_adjustment():
    _strain_scaling_test(
        [0.00068827, 0.00068036, 0.000688, 0.00068545, 0.00069104,
         0.00068033, 0.00068023, 0.00068316, 0.00067672, 0.000679],
        [-0.00014722, -0.00014553, -0.00014717, -0.00014662, -0.00014782,
         -0.00014553, -0.0001455, -0.00014613, -0.00014475, -0.00014524],
        calibration_adjustment=1.123)


def test_strain_scaling_full_bridge_type_i_with_initial_voltage():
    _strain_scaling_test(
        [0.00068827, 0.00068036, 0.000688, 0.00068545, 0.00069104,
         0.00068033, 0.00068023, 0.00068316, 0.00067672, 0.000679],
        [0.00012604, 0.00012755, 0.0001261, 0.00012658, 0.00012552,
         0.00012756, 0.00012758, 0.00012702, 0.00012824, 0.00012781],
        initial_bridge_voltage=0.00135)


def _strain_scaling_test(input_data, expected_data, initial_bridge_voltage=0.0, calibration_adjustment=1.0):
    data = StubTdmsData(np.array(input_data))
    expected_data = np.array(expected_data)

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Strain",
        "NI_Scale[0]_Strain_Configuration": 10183,
        "NI_Scale[0]_Strain_Poisson_Ratio": 0.3,
        "NI_Scale[0]_Strain_Gage_Resistance": 350.0,
        "NI_Scale[0]_Strain_Lead_Wire_Resistance": 0.0,
        "NI_Scale[0]_Strain_Initial_Bridge_Voltage": initial_bridge_voltage,
        "NI_Scale[0]_Strain_Gage_Factor": 2.1,
        "NI_Scale[0]_Strain_Bridge_Shunt_Calibration_Gain_Adjustment": calibration_adjustment,
        "NI_Scale[0]_Strain_Voltage_Excitation": 2.5,
        "NI_Scale[0]_Strain_Input_Source": 0xFFFFFFFF,
    }

    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_data)


def test_unsupported_strain_configuration():
    data = StubTdmsData(np.array([0.0, 0.0, 0.0]))

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Strain",
        "NI_Scale[0]_Strain_Configuration": 10184,
        "NI_Scale[0]_Strain_Poisson_Ratio": 0.3,
        "NI_Scale[0]_Strain_Gage_Resistance": 350.0,
        "NI_Scale[0]_Strain_Lead_Wire_Resistance": 0.0,
        "NI_Scale[0]_Strain_Initial_Bridge_Voltage": 0.0,
        "NI_Scale[0]_Strain_Gage_Factor": 2.1,
        "NI_Scale[0]_Strain_Bridge_Shunt_Calibration_Gain_Adjustment": 1.0,
        "NI_Scale[0]_Strain_Voltage_Excitation": 2.5,
        "NI_Scale[0]_Strain_Input_Source": 0xFFFFFFFF,
    }

    scaling = get_scaling(properties, {}, {})

    with pytest.raises(Exception) as exc_info:
        _ = scaling.scale(data)
    assert str(exc_info.value) == "Strain gauge configuration 10184 is not supported"


def test_multiple_scalings_applied_in_order():
    """Test all scalings applied from multiple scalings
    """

    data = StubTdmsData(np.array([1.0, 2.0, 3.0]))
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

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_multiple_scalings_but_all_with_raw_data_input():
    """Test that only the last scaling is applied from multiple scalings
       when it has the raw data as the input source
    """

    data = StubTdmsData(np.array([1.0, 2.0, 3.0]))
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

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_scaling_from_group():
    """Test linear scaling in a group"""

    data = StubTdmsData(np.array([1.0, 2.0, 3.0]))
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    group_properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0,
    }
    scaling = get_scaling({}, group_properties, {})
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_scaling_from_root():
    """Test linear scaling in the root object"""

    data = StubTdmsData(np.array([1.0, 2.0, 3.0]))
    expected_scaled_data = np.array([12.0, 14.0, 16.0])

    root_properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0,
    }
    scaling = get_scaling({}, {}, root_properties)
    scaled_data = scaling.scale(data)

    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


def test_scaling_status_scaled():
    """ When the scaling status is scaled, data is already scaled so scaling should not be applied
    """
    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": 2.0,
        "NI_Scale[0]_Linear_Y_Intercept": 10.0,
        "NI_Scaling_Status": "scaled",
    }
    scaling = get_scaling(properties, {}, {})
    assert scaling is None


def test_advanced_api_scaling():
    """Test AdvancedAPI scaling (no-op)"""

    data = StubTdmsData(np.array([1, 2, 3], dtype=np.dtype('int32')))
    expected_scaled_data = np.array([1, 2, 3])

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "AdvancedAPI",
    }
    scaling = get_scaling(properties, {}, {})
    scaled_data = scaling.scale(data)

    assert scaling.get_dtype(types.Int32, None) == np.dtype('int32')
    assert scaled_data.dtype == np.dtype('int32')
    np.testing.assert_almost_equal(scaled_data, expected_scaled_data)


class StubTdmsData(object):
    def __init__(self, data):
        self.data = data
        self.scaler_data = None


class StubDaqmxData(object):
    def __init__(self, scaler_data):
        self.data = None
        self.scaler_data = scaler_data
