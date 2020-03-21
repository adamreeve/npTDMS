from nptdms.utils import OrderedDict


def from_tdms_file(tdms_file, time_index=False, absolute_time=False):
    """
    Converts the TDMS file to a DataFrame

    :param tdms_file: TDMS file object to convert.
    :param time_index: Whether to include a time index for the dataframe.
    :param absolute_time: If time_index is true, whether the time index
        values are absolute times or relative to the start time.
    :return: The full TDMS file data.
    :rtype: pandas.DataFrame
    """

    import pandas as pd

    dataframe_dict = OrderedDict()
    for key, value in tdms_file.objects.items():
        if value.has_data:
            index = value.time_track(absolute_time) if time_index else None
            dataframe_dict[key] = pd.Series(data=value.data, index=index)
    return pd.DataFrame.from_dict(dataframe_dict)


def from_group(group, scaled_data=True):
    """
    Converts a TDMS group object to a DataFrame

    :param group: Group object to convert.
    :param scaled_data: Whether to return scaled or raw data.
    :return: The TDMS object data.
    :rtype: pandas.DataFrame
    """

    import pandas as pd

    return pd.DataFrame.from_dict(OrderedDict(
        (ch.channel, pd.Series(_get_data(ch, scaled_data)))
        for ch in group.tdms_file.group_channels(group.group)))


def from_channel(channel, absolute_time=False, scaled_data=True):
    """
    Converts the TDMS object to a DataFrame

    :param channel: Channel object to convert.
    :param absolute_time: Whether times should be absolute rather than
        relative to the start time.
    :param scaled_data: Whether to return scaled or raw data.
    :return: The TDMS object data.
    :rtype: pandas.DataFrame
    """

    import pandas as pd

    # When absolute_time is True,
    # use the wf_start_time as offset for the time_track()
    try:
        time = channel.time_track(absolute_time)
    except KeyError:
        time = None

    return pd.DataFrame(
        _get_data(channel, scaled_data), index=time, columns=[channel.path])


def _get_data(chan, scaled_data):
    if scaled_data:
        return chan.data
    else:
        return chan.raw_data
