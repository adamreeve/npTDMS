import numpy as np
import pytest
from hypothesis import (given, strategies, settings)

from nptdms import thermocouples
import thermocouples_reference


settings.register_profile("thermocouples", deadline=None, max_examples=100)
settings.load_profile("thermocouples")


reference_thermocouple_b = thermocouples_reference.thermocouples['B']
reference_thermocouple_e = thermocouples_reference.thermocouples['E']
reference_thermocouple_j = thermocouples_reference.thermocouples['J']
reference_thermocouple_k = thermocouples_reference.thermocouples['K']
reference_thermocouple_n = thermocouples_reference.thermocouples['N']
reference_thermocouple_r = thermocouples_reference.thermocouples['R']
reference_thermocouple_s = thermocouples_reference.thermocouples['S']
reference_thermocouple_t = thermocouples_reference.thermocouples['T']


def test_scale_temperature_to_voltage():
    thermocouple = thermocouples.Thermocouple(
        forward_polynomials=[
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(None, 10),
                coefficients=[0.0, 1.0]),
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(10, 20),
                coefficients=[1.0, 2.0]),
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(20, None),
                coefficients=[2.0, 3.0]),
        ],
        inverse_polynomials=[]
    )
    voltages = thermocouple.celsius_to_mv(np.array([0.0, 9.0, 10.0, 11.0, 19.0, 20.0, 21.0]))
    np.testing.assert_almost_equal(voltages, np.array([0.0, 9.0, 21.0, 23.0, 39.0, 62.0, 65.0]))


def test_scale_voltage_to_temperature():
    thermocouple = thermocouples.Thermocouple(
        forward_polynomials=[],
        inverse_polynomials=[
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(None, 10),
                coefficients=[0.0, 1.0]),
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(10, 20),
                coefficients=[1.0, 2.0]),
            thermocouples.Polynomial(
                applicable_range=thermocouples.Range(20, None),
                coefficients=[2.0, 3.0]),
        ]
    )
    temperatures = thermocouple.mv_to_celsius(np.array([0.0, 9.0, 10.0, 11.0, 19.0, 20.0, 21.0]))
    np.testing.assert_almost_equal(temperatures, np.array([0.0, 9.0, 21.0, 23.0, 39.0, 62.0, 65.0]))


def test_non_contiguous_forward_polynomial_ranges():
    with pytest.raises(ValueError) as exc_info:
        _ = thermocouples.Thermocouple(
            forward_polynomials=[
                thermocouples.Polynomial(
                    applicable_range=thermocouples.Range(None, 10),
                    coefficients=[0.0, 1.0]),
                thermocouples.Polynomial(
                    applicable_range=thermocouples.Range(11, None),
                    coefficients=[0.0, 1.0]),
            ],
            inverse_polynomials=[]
        )
    assert "Polynomial ranges must be contiguous" in str(exc_info.value)


def test_non_contiguous_inverse_polynomial_ranges():
    with pytest.raises(ValueError) as exc_info:
        _ = thermocouples.Thermocouple(
            forward_polynomials=[],
            inverse_polynomials=[
                thermocouples.Polynomial(
                    applicable_range=thermocouples.Range(None, 10),
                    coefficients=[0.0, 1.0]),
                thermocouples.Polynomial(
                    applicable_range=thermocouples.Range(11, None),
                    coefficients=[0.0, 1.0]),
            ]
        )
    assert "Polynomial ranges must be contiguous" in str(exc_info.value)


def test_no_range():
    with pytest.raises(ValueError) as exc_info:
        _ = thermocouples.Range(None, None)
    assert "At least one of start and end must be provided" in str(exc_info.value)


@pytest.mark.parametrize("start,end", [(2.0, 1.0), (1.0, 1.0)])
def test_invalid_range(start, end):
    with pytest.raises(ValueError) as exc_info:
        _ = thermocouples.Range(start, end)
    assert "start must be less than end" in str(exc_info.value)


@given(temperature=strategies.floats(0.0, 1820.0))
def test_type_b_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_b, thermocouples.type_b, temperature)


@given(voltage=strategies.floats(0.291, 13.820))
def test_type_b_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_b, thermocouples.type_b, voltage, max_error=0.03)


@given(temperature=strategies.floats(-270.0, 1000.0))
def test_type_e_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_e, thermocouples.type_e, temperature)


@given(voltage=strategies.floats(-8.825, 76.373))
def test_type_e_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_e, thermocouples.type_e, voltage, max_error=0.03)


@given(temperature=strategies.floats(-210.0, 1200.0))
def test_type_j_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_j, thermocouples.type_j, temperature)


@given(voltage=strategies.floats(-8.095, 69.553))
def test_type_j_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_j, thermocouples.type_j, voltage, max_error=0.05)


@given(temperature=strategies.floats(-270.000, 1372.000))
def test_type_k_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_k, thermocouples.type_k, temperature)


@given(voltage=strategies.floats(-5.891, 54.886))
def test_type_k_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_k, thermocouples.type_k, voltage, max_error=0.06)


@given(temperature=strategies.floats(-270.000, 1300.0))
def test_type_n_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_n, thermocouples.type_n, temperature)


@given(voltage=strategies.floats(-3.990, 47.513))
def test_type_n_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_n, thermocouples.type_n, voltage, max_error=0.04)


@given(temperature=strategies.floats(-50.000, 1768.1))
def test_type_r_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_r, thermocouples.type_r, temperature)


@given(voltage=strategies.floats(-0.226, 21.103))
def test_type_r_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_r, thermocouples.type_r, voltage, max_error=0.02)


@given(temperature=strategies.floats(-50.000, 1768.1))
def test_type_s_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_s, thermocouples.type_s, temperature)


@given(voltage=strategies.floats(-0.235, 18.693))
def test_type_s_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_s, thermocouples.type_s, voltage, max_error=0.02)


@given(temperature=strategies.floats(-270.000, 400.000))
def test_type_t_temperature_to_voltage(temperature):
    _test_temperature_to_voltage(reference_thermocouple_t, thermocouples.type_t, temperature)


@given(voltage=strategies.floats(-5.603, 20.872))
def test_type_t_voltage_to_temperature(voltage):
    _test_voltage_to_temperature(reference_thermocouple_t, thermocouples.type_t, voltage, max_error=0.04)


def _test_temperature_to_voltage(reference_thermocouple, thermocouple, temperature):
    reference_voltage = reference_thermocouple.emf_mVC(temperature, Tref=0.0)
    voltage = thermocouple.celsius_to_mv(temperature)
    assert abs(voltage - reference_voltage) < 1.0E-6


def _test_voltage_to_temperature(reference_thermocouple, thermocouple, voltage, max_error):
    reference_temperature = reference_thermocouple.inverse_CmV(voltage, Tref=0.0)
    temperature = thermocouple.mv_to_celsius(voltage)
    assert abs(temperature - reference_temperature) < max_error + 1.0E-6
