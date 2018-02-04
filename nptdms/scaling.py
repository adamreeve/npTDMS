import logging
import numpy as np
import re


log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class LinearScaling(object):
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope

    def scale(self, data):
        return data * self.slope + self.intercept


class PolynomialScaling(object):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def scale(self, data):
        return np.polynomial.polynomial.polyval(data, self.coefficients)


def get_scaling(channel):
    scalings = (_get_object_scaling(o) for o in _tdms_hierarchy(channel))
    try:
        return next(s for s in scalings if s is not None)
    except StopIteration:
        return None


def _get_object_scaling(obj):
    scale_index = _get_scale_index(obj.properties)
    if scale_index is None:
        return None

    scale_type = obj.properties['NI_Scale[%d]_Scale_Type' % scale_index]
    if scale_type == 'Polynomial':
        try:
            number_of_coefficients = obj.properties[
                'NI_Scale[%d]_Polynomial_Coefficients_Size' % (scale_index)]
        except KeyError:
            number_of_coefficients = 4
        coefficients = [
            obj.properties[
                'NI_Scale[%d]_Polynomial_Coefficients[%d]' % (scale_index, i)]
            for i in range(number_of_coefficients)]
        return PolynomialScaling(coefficients)
    elif scale_type == 'Linear':
        return LinearScaling(
            obj.properties["NI_Scale[%d]_Linear_Y_Intercept" % scale_index],
            obj.properties["NI_Scale[%d]_Linear_Slope" % scale_index])
    else:
        log.warning("Unsupported scale type: %s", scale_type)
        return None


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


def _get_scale_index(properties):
    matches = (_scale_regex.match(key) for key in properties.keys())
    try:
        return max(int(m.group(1)) for m in matches if m is not None)
    except ValueError:
        return None
