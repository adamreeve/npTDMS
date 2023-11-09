from collections import OrderedDict
import numpy as np


def from_tdms_file(tdms_file, time_index=False, absolute_time=False, scaled_data=True,
                   arrow_dtypes=False):
    """
    Converts the TDMS file to a DataFrame. DataFrame columns are named using the TDMS object paths.

    :param tdms_file: TDMS file object to convert.
    :param time_index: Whether to include a time index for the dataframe.
    :param absolute_time: If time_index is true, whether the time index
        values are absolute times or relative to the start time.
    :param scaled_data: By default the scaled data will be used.
        Set to False to use raw unscaled data.
    :param arrow_dtypes: Use PyArrow data types in the DataFrame.
    :return: The full TDMS file data.
    :rtype: pandas.DataFrame
    """

    channels_to_export = OrderedDict()
    for group in tdms_file.groups():
        for channel in group.channels():
            channels_to_export[channel.path] = channel
    return _channels_to_dataframe(
        channels_to_export,
        time_index=time_index, absolute_time=absolute_time, scaled_data=scaled_data,
        arrow_dtypes=arrow_dtypes)


def from_group(group, time_index=False, absolute_time=False, scaled_data=True, arrow_dtypes=False):
    """
    Converts a TDMS group object to a DataFrame. DataFrame columns are named using the channel names.

    :param group: Group object to convert.
    :param time_index: Whether to include a time index for the dataframe.
    :param absolute_time: If time_index is true, whether the time index
        values are absolute times or relative to the start time.
    :param scaled_data: By default the scaled data will be used.
        Set to False to use raw unscaled data.
    :param arrow_dtypes: Use PyArrow data types in the DataFrame.
    :return: The TDMS object data.
    :rtype: pandas.DataFrame
    """

    channels_to_export = OrderedDict((ch.name, ch) for ch in group.channels())
    return _channels_to_dataframe(
        channels_to_export, time_index=time_index, absolute_time=absolute_time,
        scaled_data=scaled_data, arrow_dtypes=arrow_dtypes)


def from_channel(channel, time_index=False, absolute_time=False, scaled_data=True,
                 arrow_dtypes=False):
    """
    Converts the TDMS channel to a DataFrame

    :param channel: Channel object to convert.
    :param time_index: Whether to include a time index for the dataframe.
    :param absolute_time: If time_index is true, whether the time index
        values are absolute times or relative to the start time.
    :param scaled_data: By default the scaled data will be used.
        Set to False to use raw unscaled data.
    :param arrow_dtypes: Use PyArrow data types in the DataFrame.
    :return: The TDMS object data.
    :rtype: pandas.DataFrame
    """

    channels_to_export = {channel.path: channel}
    return _channels_to_dataframe(
        channels_to_export, time_index=time_index, absolute_time=absolute_time,
        scaled_data=scaled_data, arrow_dtypes=arrow_dtypes)


def _channels_to_dataframe(
        channels_to_export, time_index=False, absolute_time=False, scaled_data=True,
        arrow_dtypes=False):
    import pandas as pd

    column_data = []
    for column_name, channel in channels_to_export.items():
        index = channel.time_track(absolute_time) if time_index else None
        if scaled_data:
            column_data.append((column_name, _array_for_pd(channel[:]), index))
        elif channel.scaler_data_types:
            # Channel has DAQmx raw data
            raw_data = channel.read_data(scaled=False)
            for scale_id, scaler_data in raw_data.items():
                scaler_column_name = column_name + "[{0:d}]".format(scale_id)
                column_data.append((scaler_column_name, scaler_data, index))
        else:
            # Raw data for normal TDMS file
            raw_data = channel.read_data(scaled=False)
            column_data.append((column_name, _array_for_pd(raw_data), index))

    dataframe_dict = OrderedDict()
    if arrow_dtypes:
        import pyarrow as pa

        for column_name, data, index in column_data:
            # Let arrow deduce data types from the numpy dtypes
            if index is not None:
                index_array = pa.array(index)
                index = pd.Index(index_array, dtype=pd.ArrowDtype(index_array.type))
            data_array = pa.array(data)
            dataframe_dict[column_name] = pd.Series(
                data=data_array, dtype=pd.ArrowDtype(data_array.type), index=index)
    else:
        for column_name, data, index in column_data:
            dataframe_dict[column_name] = pd.Series(data=data, index=index)

    return pd.DataFrame.from_dict(dataframe_dict)


def _array_for_pd(array):
    """ Convert data array to a format suitable for a Pandas dataframe
    """
    if np.issubdtype(array.dtype, np.dtype('void')) and len(array.dtype) == 0:
        # If dtype is void then the array must also be empty.
        # Pandas doesn't like void data types, so these are converted to empty float64 arrays
        # and Pandas will fill values with NaN
        return np.empty(0, dtype='float64')
    return array
