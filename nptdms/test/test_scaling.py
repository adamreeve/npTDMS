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


_strain_voltage = [
        0.0068827, 0.0068036, 0.00688, 0.0068545, 0.0069104,
        0.0068033, 0.0068023, 0.0068316, 0.0067672, 0.00679]

_measured_strain = {
    10183: {
        'baseline': [
            -1.310990e-03, -1.295924e-03, -1.310476e-03, -1.305619e-03, -1.316267e-03,
            -1.295867e-03, -1.295676e-03, -1.301257e-03, -1.288990e-03, -1.293333e-03,
        ],
        'with_adjustment': [
            -1.472242e-03, -1.455322e-03, -1.471665e-03, -1.466210e-03, -1.478167e-03,
            -1.455258e-03, -1.455044e-03, -1.461312e-03, -1.447536e-03, -1.452413e-03,
        ],
        'with_initial_voltage': [
            -1.053848e-03, -1.038781e-03, -1.053333e-03, -1.048476e-03, -1.059124e-03,
            -1.038724e-03, -1.038533e-03, -1.044114e-03, -1.031848e-03, -1.036190e-03,
        ],
        'with_lead_resistance': [
            -1.310990e-03, -1.295924e-03, -1.310476e-03, -1.305619e-03, -1.316267e-03,
            -1.295867e-03, -1.295676e-03, -1.301257e-03, -1.288990e-03, -1.293333e-03,
        ],
        'with_all': [
            -1.183471e-03, -1.166551e-03, -1.182893e-03, -1.177439e-03, -1.189396e-03,
            -1.166487e-03, -1.166273e-03, -1.172540e-03, -1.158765e-03, -1.163642e-03,
        ],
    },
    10184: {
        'baseline': [
            -2.016908e-03, -1.993729e-03, -2.016117e-03, -2.008645e-03, -2.025026e-03,
            -1.993641e-03, -1.993348e-03, -2.001934e-03, -1.983062e-03, -1.989744e-03,
        ],
        'with_adjustment': [
            -2.264988e-03, -2.238958e-03, -2.264100e-03, -2.255708e-03, -2.274104e-03,
            -2.238859e-03, -2.238530e-03, -2.248172e-03, -2.226979e-03, -2.234482e-03,
        ],
        'with_initial_voltage': [
            -1.621304e-03, -1.598125e-03, -1.620513e-03, -1.613040e-03, -1.629421e-03,
            -1.598037e-03, -1.597744e-03, -1.606330e-03, -1.587458e-03, -1.594139e-03,
        ],
        'with_lead_resistance': [
            -2.016908e-03, -1.993729e-03, -2.016117e-03, -2.008645e-03, -2.025026e-03,
            -1.993641e-03, -1.993348e-03, -2.001934e-03, -1.983062e-03, -1.989744e-03,
        ],
        'with_all': [
            -1.820724e-03, -1.794694e-03, -1.819836e-03, -1.811444e-03, -1.829840e-03,
            -1.794595e-03, -1.794266e-03, -1.803908e-03, -1.782715e-03, -1.790218e-03,
        ],
    },
    10185: {
        'baseline': [
            -2.013923e-03, -1.990812e-03, -2.013134e-03, -2.005684e-03, -2.022016e-03,
            -1.990724e-03, -1.990432e-03, -1.998993e-03, -1.980176e-03, -1.986838e-03,
        ],
        'with_adjustment': [
            -2.261635e-03, -2.235681e-03, -2.260750e-03, -2.252383e-03, -2.270724e-03,
            -2.235583e-03, -2.235255e-03, -2.244869e-03, -2.223738e-03, -2.231219e-03,
        ],
        'with_initial_voltage': [
            -1.619374e-03, -1.596250e-03, -1.618585e-03, -1.611130e-03, -1.627472e-03,
            -1.596162e-03, -1.595869e-03, -1.604435e-03, -1.585608e-03, -1.592274e-03,
        ],
        'with_lead_resistance': [
            -2.013923e-03, -1.990812e-03, -2.013134e-03, -2.005684e-03, -2.022016e-03,
            -1.990724e-03, -1.990432e-03, -1.998993e-03, -1.980176e-03, -1.986838e-03,
        ],
        'with_all': [
            -1.818557e-03, -1.792588e-03, -1.817671e-03, -1.809299e-03, -1.827651e-03,
            -1.792490e-03, -1.792161e-03, -1.801781e-03, -1.780638e-03, -1.788123e-03,
        ],
    },
    10188: {
        'baseline': [
            -4.021893e-03, -3.975806e-03, -4.020319e-03, -4.005462e-03, -4.038031e-03,
            -3.975631e-03, -3.975048e-03, -3.992120e-03, -3.954596e-03, -3.967881e-03,
        ],
        'with_adjustment': [
            -4.516585e-03, -4.464830e-03, -4.514819e-03, -4.498134e-03, -4.534709e-03,
            -4.464633e-03, -4.463979e-03, -4.483151e-03, -4.441012e-03, -4.455931e-03,
        ],
        'with_initial_voltage': [
            -3.234898e-03, -3.188758e-03, -3.233323e-03, -3.218449e-03, -3.251055e-03,
            -3.188583e-03, -3.188000e-03, -3.205091e-03, -3.167524e-03, -3.180824e-03,
        ],
        'with_lead_resistance': [
            -4.036073e-03, -3.989823e-03, -4.034494e-03, -4.019585e-03, -4.052268e-03,
            -3.989648e-03, -3.989063e-03, -4.006195e-03, -3.968539e-03, -3.981871e-03,
        ],
        'with_all': [
            -3.645599e-03, -3.593601e-03, -3.643824e-03, -3.627061e-03, -3.663807e-03,
            -3.593403e-03, -3.592746e-03, -3.612008e-03, -3.569671e-03, -3.584660e-03,
        ],
    },
    10189: {
        'baseline': [
            -2.621981e-03, -2.591848e-03, -2.620952e-03, -2.611238e-03, -2.632533e-03,
            -2.591733e-03, -2.591352e-03, -2.602514e-03, -2.577981e-03, -2.586667e-03,
        ],
        'with_adjustment': [
            -2.944485e-03, -2.910645e-03, -2.943330e-03, -2.932420e-03, -2.956335e-03,
            -2.910517e-03, -2.910089e-03, -2.922624e-03, -2.895073e-03, -2.904827e-03,
        ],
        'with_initial_voltage': [
            -2.107695e-03, -2.077562e-03, -2.106667e-03, -2.096952e-03, -2.118248e-03,
            -2.077448e-03, -2.077067e-03, -2.088229e-03, -2.063695e-03, -2.072381e-03,
        ],
        'with_lead_resistance': [
            -2.631225e-03, -2.600986e-03, -2.630193e-03, -2.620445e-03, -2.641815e-03,
            -2.600871e-03, -2.600489e-03, -2.611690e-03, -2.587070e-03, -2.595787e-03,
        ],
        'with_all': [
            -2.375287e-03, -2.341328e-03, -2.374128e-03, -2.363180e-03, -2.387179e-03,
            -2.341199e-03, -2.340770e-03, -2.353349e-03, -2.325701e-03, -2.335489e-03,
        ],
    },
    10271: {
        'baseline': [
            -5.215246e-03, -5.155634e-03, -5.213211e-03, -5.193994e-03, -5.236120e-03,
            -5.155408e-03, -5.154654e-03, -5.176736e-03, -5.128199e-03, -5.145384e-03,
        ],
        'with_adjustment': [
            -5.856721e-03, -5.789777e-03, -5.854436e-03, -5.832856e-03, -5.880162e-03,
            -5.789523e-03, -5.788676e-03, -5.813475e-03, -5.758968e-03, -5.778266e-03,
        ],
        'with_initial_voltage': [
            -4.196815e-03, -4.137074e-03, -4.194776e-03, -4.175517e-03, -4.217733e-03,
            -4.136848e-03, -4.136092e-03, -4.158222e-03, -4.109581e-03, -4.126802e-03,
        ],
        'with_lead_resistance': [
            -5.233633e-03, -5.173811e-03, -5.231592e-03, -5.212307e-03, -5.254581e-03,
            -5.173584e-03, -5.172828e-03, -5.194988e-03, -5.146280e-03, -5.163525e-03,
        ],
        'with_all': [
            -4.729640e-03, -4.662315e-03, -4.727342e-03, -4.705639e-03, -4.753214e-03,
            -4.662059e-03, -4.661208e-03, -4.686147e-03, -4.631330e-03, -4.650738e-03,
        ],
    },
    10272: {
        'baseline': [
            -5.215246e-03, -5.155634e-03, -5.213211e-03, -5.193994e-03, -5.236120e-03,
            -5.155408e-03, -5.154654e-03, -5.176736e-03, -5.128199e-03, -5.145384e-03,
        ],
        'with_adjustment': [
            -5.856721e-03, -5.789777e-03, -5.854436e-03, -5.832856e-03, -5.880162e-03,
            -5.789523e-03, -5.788676e-03, -5.813475e-03, -5.758968e-03, -5.778266e-03,
        ],
        'with_initial_voltage': [
            -4.196815e-03, -4.137074e-03, -4.194776e-03, -4.175517e-03, -4.217733e-03,
            -4.136848e-03, -4.136092e-03, -4.158222e-03, -4.109581e-03, -4.126802e-03,
        ],
        'with_lead_resistance': [
            -5.233633e-03, -5.173811e-03, -5.231592e-03, -5.212307e-03, -5.254581e-03,
            -5.173584e-03, -5.172828e-03, -5.194988e-03, -5.146280e-03, -5.163525e-03,
        ],
        'with_all': [
            -4.729640e-03, -4.662315e-03, -4.727342e-03, -4.705639e-03, -4.753214e-03,
            -4.662059e-03, -4.661208e-03, -4.686147e-03, -4.631330e-03, -4.650738e-03,
        ],
    },
}


@pytest.mark.parametrize("configuration", [
    10183, 10184, 10185, 10188, 10189, 10271, 10272])
@pytest.mark.parametrize("key", [
    "baseline", "with_adjustment", "with_initial_voltage", "with_lead_resistance", "with_all"])
def test_strain_scaling(configuration, key):
    data = StubTdmsData(np.array(_strain_voltage))
    expected_data = np.array(_measured_strain[configuration][key])

    lead_wire_resistance = 0.0
    initial_bridge_voltage = 0.0
    gain_adjustment = 1.0

    if key in ("with_adjustment", "with_all"):
        gain_adjustment = 1.123
    if key in ("with_initial_voltage", "with_all"):
        initial_bridge_voltage = 0.00135
    if key in ("with_lead_resistance", "with_all"):
        lead_wire_resistance = 1.234

    properties = {
        "NI_Number_Of_Scales": 1,
        "NI_Scale[0]_Scale_Type": "Strain",
        "NI_Scale[0]_Strain_Configuration": configuration,
        "NI_Scale[0]_Strain_Poisson_Ratio": 0.3,
        "NI_Scale[0]_Strain_Gage_Resistance": 350.0,
        "NI_Scale[0]_Strain_Lead_Wire_Resistance": lead_wire_resistance,
        "NI_Scale[0]_Strain_Initial_Bridge_Voltage": initial_bridge_voltage,
        "NI_Scale[0]_Strain_Gage_Factor": 2.1,
        "NI_Scale[0]_Strain_Bridge_Shunt_Calibration_Gain_Adjustment": gain_adjustment,
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
        "NI_Scale[0]_Strain_Configuration": 12345,
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
    assert str(exc_info.value) == "Strain gauge configuration 12345 is not supported"


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
