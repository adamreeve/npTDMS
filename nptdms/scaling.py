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

    def scale(self, data):
        return data * self.slope + self.intercept


class PolynomialScaling(object):
    """ Polynomial scaling with an arbitrary number of coefficients
    """
    def __init__(self, coefficients, input_source):
        self.coefficients = coefficients
        self.input_source = input_source

    def scale(self, data):
        return np.polynomial.polynomial.polyval(data, self.coefficients)


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
        final_scale = self.scalings[-1]
        return self._compute_scaled_data(final_scale, data, {})

    def scale_daqmx(self, scaler_data):
        final_scale = self.scalings[-1]
        return self._compute_scaled_data(final_scale, None, scaler_data)

    def _compute_scaled_data(self, scaling, raw_data, scaler_data):
        """ Compute output data from a single scale in the set of all scalings,
            computing any required input scales recursively.
        """
        if scaling.input_source == RAW_DATA_INPUT_SOURCE:
            if raw_data is None:
                raise Exception("Invalid scaling input source for DAQmx data")
            return scaling.scale(raw_data)

        input_scaling = self.scalings[scaling.input_source]
        if input_scaling is None:
            raise Exception(
                "Cannot compute data for scale %d" % scaling.input_source)
        elif isinstance(input_scaling, DaqMxScalerScaling):
            input_data = input_scaling.scale_daqmx(scaler_data)
        else:
            input_data = self._compute_scaled_data(
                input_scaling, raw_data, scaler_data)
        return scaling.scale(input_data)


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
            scalings[scale_index] = _read_polynomial_scaling(obj, scale_index)
        elif scale_type == 'Linear':
            scalings[scale_index] = _read_linear_scaling(obj, scale_index)
        else:
            log.warning("Unsupported scale type: %s", scale_type)

    if not scalings:
        return None
    if len(scalings) > 1:
        return MultiScaling(scalings)
    return scalings[0]


def _read_linear_scaling(obj, scale_index):
    try:
        input_source = obj.properties[
            "NI_Scale[%d]_Linear_Input_Source" % scale_index]
    except KeyError:
        input_source = RAW_DATA_INPUT_SOURCE
    return LinearScaling(
        obj.properties["NI_Scale[%d]_Linear_Y_Intercept" % scale_index],
        obj.properties["NI_Scale[%d]_Linear_Slope" % scale_index],
        input_source)


def _read_polynomial_scaling(obj, scale_index):
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
