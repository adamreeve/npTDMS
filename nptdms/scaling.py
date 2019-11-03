import numpy as np
import re

from nptdms.log import log_manager


log = log_manager.get_logger(__name__)

RAW_DATA_INPUT_SOURCE = 0xFFFFFFFF


class LinearScaling(object):
    """ Linear scaling with slope and intercept
    """
    def __init__(self, intercept, slope, input_source):
        self.intercept = intercept
        self.slope = slope
        self.input_source = input_source

    @staticmethod
    def from_object(obj, scale_index):
        try:
            input_source = obj.properties[
                "NI_Scale[%d]_Linear_Input_Source" % scale_index]
        except KeyError:
            input_source = RAW_DATA_INPUT_SOURCE
        return LinearScaling(
            obj.properties["NI_Scale[%d]_Linear_Y_Intercept" % scale_index],
            obj.properties["NI_Scale[%d]_Linear_Slope" % scale_index],
            input_source)

    def scale(self, data):
        return data * self.slope + self.intercept


class PolynomialScaling(object):
    """ Polynomial scaling with an arbitrary number of coefficients
    """
    def __init__(self, coefficients, input_source):
        self.coefficients = coefficients
        self.input_source = input_source

    @staticmethod
    def from_object(obj, scale_index):
        try:
            number_of_coefficients = obj.properties[
                'NI_Scale[%d]_Polynomial_Coefficients_Size' % (scale_index)]
        except KeyError:
            number_of_coefficients = 4
        try:
            input_source = obj.properties[
                "NI_Scale[%d]_Polynomial_Input_Source" % scale_index]
        except KeyError:
            input_source = RAW_DATA_INPUT_SOURCE
        coefficients = [
            obj.properties[
                'NI_Scale[%d]_Polynomial_Coefficients[%d]' % (scale_index, i)]
            for i in range(number_of_coefficients)]
        return PolynomialScaling(coefficients, input_source)

    def scale(self, data):
        return np.polynomial.polynomial.polyval(data, self.coefficients)


class RtdScaling(object):
    """ Converts a signal from a resitance temperature detector into
        degrees celcius using the Callendar-Van Dusen equation
    """
    def __init__(
            self, current_excitation, r0_nominal_resistance,
            a, b, c,
            lead_wire_resistance, resistance_configuration, input_source):
        self.current_excitation = current_excitation
        self.r0_nominal_resistance = r0_nominal_resistance
        self.a = a
        self.b = b
        self.c = c
        self.lead_wire_resistance = lead_wire_resistance
        self.resistance_configuration = resistance_configuration
        self.input_source = input_source

    @staticmethod
    def from_object(obj, scale_index):
        prefix = "NI_Scale[%d]" % scale_index
        current_excitation = obj.properties[
            "%s_RTD_Current_Excitation" % prefix]
        r0_nominal_resistance = obj.properties[
            "%s_RTD_R0_Nominal_Resistance" % prefix]
        a = obj.properties["%s_RTD_A" % prefix]
        b = obj.properties["%s_RTD_B" % prefix]
        c = obj.properties["%s_RTD_C" % prefix]
        lead_wire_resistance = obj.properties[
            "%s_RTD_Lead_Wire_Resistance" % prefix]
        resistance_configuration = obj.properties[
            "%s_RTD_Resistance_Configuration" % prefix]
        input_source = obj.properties[
            "%s_RTD_Input_Source" % prefix]
        return RtdScaling(
            current_excitation, r0_nominal_resistance, a, b, c,
            lead_wire_resistance, resistance_configuration, input_source)

    def scale(self, data):
        """ Convert voltage data to temperature
        """
        r_0 = self.r0_nominal_resistance
        a = self.a
        b = self.b

        # R(T) = R(0)[1 + A*T + B*T^2 + (T - 100)*C*T^3]
        # R(T) = V/I

        if self.lead_wire_resistance != 0.0:
            # This is untested so throw an error.
            # For 3 & 4 lead wire configuration the lead resistance should not
            # be needed, but for configuration 2, we should have
            # R_t = V/I - lead resistance
            raise NotImplementedError(
                "RTD scaling with non-zero lead wire resistance "
                "is not implemented")

        r_t = data / self.current_excitation

        if np.all(r_t >= self.r0_nominal_resistance):
            # For R(T) > R(0), temperature is positive and we can use the
            # quadratic form of the equation without C:
            return ((-a + np.sqrt(a ** 2 - 4.0 * b * (1.0 - r_t / r_0))) /
                    (2.0 * self.b))
        else:
            # We would need to solve for the roots of the full quartic equation
            # for any cases where R(T) < R(0), and work out which root is the
            # correct solution. For R(T) > R(0) we set C to zero.
            raise NotImplementedError(
                "RTD scaling for temperatures < 0 is not implemented")


class TableScaling(object):
    """ Scales data using a map from input to output values with
        linear interpolation for points in between inputs.
    """
    def __init__(
            self, pre_scaled_values, scaled_values, input_source):

        # This is a bit counterintuitive but the scaled values are the input
        # values and the pre-scaled values are the output values for
        # interpolation.

        # Ensure values are monotonically increasing for interpolation to work
        if not np.all(np.diff(scaled_values) > 0):
            scaled_values = np.flip(scaled_values)
            pre_scaled_values = np.flip(pre_scaled_values)
        if not np.all(np.diff(scaled_values) > 0):
            # Reversing didn't help
            raise ValueError(
                "Table scaled values must be monotonically "
                "increasing or decreasing")

        self.input_values = scaled_values
        self.output_values = pre_scaled_values
        self.input_source = input_source

    @staticmethod
    def from_object(obj, scale_index):
        prefix = "NI_Scale[%d]_Table_" % scale_index
        try:
            input_source = obj.properties[prefix + "Input_Source"]
        except KeyError:
            input_source = RAW_DATA_INPUT_SOURCE
        num_pre_scaled_values = obj.properties[
            prefix + "Pre_Scaled_Values_Size"]
        num_scaled_values = obj.properties[
            prefix + "Scaled_Values_Size"]
        if num_pre_scaled_values != num_scaled_values:
            raise ValueError(
                "Number of pre-scaled values does not match "
                "number of scaled values")
        pre_scaled_values = np.array([
            obj.properties[prefix + "Pre_Scaled_Values[%d]" % i]
            for i in range(num_pre_scaled_values)])
        scaled_values = np.array([
            obj.properties[prefix + "Scaled_Values[%d]" % i]
            for i in range(num_scaled_values)])
        return TableScaling(pre_scaled_values, scaled_values, input_source)

    def scale(self, data):
        """ Calculate scaled data
        """
        return np.interp(data, self.input_values, self.output_values)


class ThermocoupleScaling(object):
    """ Convert between voltages in uV and degrees celcius for a Thermocouple.
        Can convert in either direction depending on the scaling direction
        parameter.
    """
    def __init__(self, type_code, scaling_direction, input_source):
        from thermocouples_reference import thermocouples

        # Thermocouple types from
        # http://zone.ni.com/reference/en-XX/help/371361R-01/glang/tdms_create_scalinginfo/#instance2
        thermocouple_type = {
            10047: 'B',
            10055: 'E',
            10072: 'J',
            10073: 'K',
            10077: 'N',
            10082: 'R',
            10085: 'S',
            10086: 'T',
        }[type_code]
        self.thermocouple = thermocouples[thermocouple_type]

        self.scaling_direction = scaling_direction
        self.input_source = input_source

    @staticmethod
    def from_object(obj, scale_index):
        prefix = "NI_Scale[%d]_Thermocouple" % scale_index
        input_source = obj.properties.get(
            "%s_Input_Source" % prefix, RAW_DATA_INPUT_SOURCE)
        type_code = obj.properties.get(
            "%s_Thermocouple_Type" % prefix, 10072)
        scaling_direction = obj.properties.get(
            "%s_Scaling_Direction" % prefix, 0)
        return ThermocoupleScaling(type_code, scaling_direction, input_source)

    def scale(self, data):
        """ Apply thermocouple scaling
        """

        # Note that the thermocouples_reference package uses mV for voltages,
        # but TDMS uses uV.
        nan = float('nan')

        def scale_uv_to_c(micro_volts):
            """Convert micro volts to degrees celcius"""
            milli_volts = micro_volts / 1000.0
            try:
                return self.thermocouple.inverse_CmV(milli_volts, Tref=0.0)
            except ValueError:
                return nan

        def scale_c_to_uv(temp):
            """Convert degrees celcius to micro volts"""
            try:
                return 1000.0 * self.thermocouple.emf_mVC(temp, Tref=0.0)
            except ValueError:
                return nan

        if self.scaling_direction == 1:
            scaled = np.vectorize(scale_c_to_uv)(data)
        else:
            scaled = np.vectorize(scale_uv_to_c)(data)
        return scaled


class AddScaling(object):
    """ Adds two scalings
    """
    def __init__(self, left_input_source, right_input_source):
        self.left_input_source = left_input_source
        self.right_input_source = right_input_source

    @staticmethod
    def from_object(obj, scale_index):
        left_input_source = obj.properties[
            "NI_Scale[%d]_Add_Left_Operand_Input_Source" % scale_index]
        right_input_source = obj.properties[
            "NI_Scale[%d]_Add_Right_Operand_Input_Source" % scale_index]
        return AddScaling(left_input_source, right_input_source)

    def scale(self, left_data, right_data):
        return left_data + right_data


class SubtractScaling(object):
    """ Subtracts one scaling from another
    """
    def __init__(self, left_input_source, right_input_source):
        self.left_input_source = left_input_source
        self.right_input_source = right_input_source

    @staticmethod
    def from_object(obj, scale_index):
        left_input_source = obj.properties[
            "NI_Scale[%d]_Subtract_Left_Operand_Input_Source" % scale_index]
        right_input_source = obj.properties[
            "NI_Scale[%d]_Subtract_Right_Operand_Input_Source" % scale_index]
        return SubtractScaling(left_input_source, right_input_source)

    def scale(self, left_data, right_data):
        """ Calculate scaled data
        """

        # Subtracting the left operand from the right doesn't make much sense,
        # but this does match the Excel TDMS plugin behaviour.
        return right_data - left_data


class DaqMxScalerScaling(object):
    """ Reads scaler from DAQmx data
    """
    def __init__(self, scale_id):
        self.scale_id = scale_id

    def scale_daqmx(self, scaler_data):
        return scaler_data[self.scale_id]


class MultiScaling(object):
    """ Computes scaled data from multiple scalings
    """
    def __init__(self, scalings):
        self.scalings = scalings

    def scale(self, data):
        final_scale = len(self.scalings) - 1
        return self._compute_scaled_data(final_scale, data, {})

    def scale_daqmx(self, scaler_data):
        final_scale = len(self.scalings) - 1
        return self._compute_scaled_data(final_scale, None, scaler_data)

    def _compute_scaled_data(self, scale_index, raw_data, scaler_data):
        """ Compute output data from a single scale in the set of all scalings,
            computing any required input scales recursively.
        """
        if scale_index == RAW_DATA_INPUT_SOURCE:
            if raw_data is None:
                raise Exception("Invalid scaling input source for DAQmx data")
            return raw_data

        scaling = self.scalings[scale_index]
        if scaling is None:
            raise Exception(
                "Cannot compute data for scale %d" % scale_index)

        if isinstance(scaling, DaqMxScalerScaling):
            return scaling.scale_daqmx(scaler_data)
        elif hasattr(scaling, 'input_source'):
            input_data = self._compute_scaled_data(
                scaling.input_source, raw_data, scaler_data)
            return scaling.scale(input_data)
        elif (hasattr(scaling, 'left_input_source') and
              hasattr(scaling, 'right_input_source')):
            left_input_data = self._compute_scaled_data(
                scaling.left_input_source, raw_data, scaler_data)
            right_input_data = self._compute_scaled_data(
                scaling.right_input_source, raw_data, scaler_data)
            return scaling.scale(left_input_data, right_input_data)
        else:
            raise ValueError("Cannot compute scaled data for %r" % scaling)


def get_scaling(channel):
    """ Get scaling for a channel from either the channel itself,
        its group, or the whole TDMS file
    """
    scalings = (_get_object_scaling(o) for o in _tdms_hierarchy(channel))
    try:
        return next(s for s in scalings if s is not None)
    except StopIteration:
        return None


def _get_object_scaling(obj):
    num_scalings = _get_number_of_scalings(obj.properties)
    if num_scalings is None or num_scalings == 0:
        return None

    scalings = [None] * num_scalings
    for scale_index in range(num_scalings):
        type_property = 'NI_Scale[%d]_Scale_Type' % scale_index
        try:
            scale_type = obj.properties[type_property]
        except KeyError:
            # Scalings are not in properties if they come from DAQmx scalers
            scalings[scale_index] = DaqMxScalerScaling(scale_index)
            continue
        if scale_type == 'Polynomial':
            scalings[scale_index] = PolynomialScaling.from_object(
                obj, scale_index)
        elif scale_type == 'Linear':
            scalings[scale_index] = LinearScaling.from_object(obj, scale_index)
        elif scale_type == 'RTD':
            scalings[scale_index] = RtdScaling.from_object(obj, scale_index)
        elif scale_type == 'Table':
            scalings[scale_index] = TableScaling.from_object(obj, scale_index)
        elif scale_type == 'Thermocouple':
            scalings[scale_index] = ThermocoupleScaling.from_object(
                obj, scale_index)
        elif scale_type == 'Add':
            scalings[scale_index] = AddScaling.from_object(obj, scale_index)
        elif scale_type == 'Subtract':
            scalings[scale_index] = SubtractScaling.from_object(
                obj, scale_index)
        else:
            log.warning("Unsupported scale type: %s", scale_type)

    if not scalings:
        return None
    if len(scalings) > 1:
        return MultiScaling(scalings)
    return scalings[0]


def _tdms_hierarchy(tdms_channel):
    yield tdms_channel

    tdms_file = tdms_channel.tdms_file
    if tdms_file is None:
        return

    group_name = tdms_channel.group
    if group_name is not None:
        try:
            yield tdms_file.object(group_name)
        except KeyError:
            pass

    try:
        yield tdms_file.object()
    except KeyError:
        pass


_scale_regex = re.compile(r"NI_Scale\[(\d+)\]_Scale_Type")


def _get_number_of_scalings(properties):
    num_scalings_property = "NI_Number_Of_Scales"
    if num_scalings_property in properties:
        return int(properties[num_scalings_property])

    matches = (_scale_regex.match(key) for key in properties.keys())
    try:
        return max(int(m.group(1)) for m in matches if m is not None) + 1
    except ValueError:
        return None
