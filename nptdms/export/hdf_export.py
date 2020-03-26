import numpy as np
from nptdms import types


def from_tdms_file(tdms_file, filepath, mode='w', group='/'):
    """
    Converts the TDMS file into an HDF5 file

    :param tdms_file: The TDMS file object to convert.
    :param filepath: The path of the HDF5 file you want to write to.
    :param mode: The write mode of the HDF5 file. This can be 'w' or 'a'
    :param group: A group in the HDF5 file that will contain the TDMS data.
    """
    import h5py

    # Groups in TDMS are mapped to the first level of the HDF5 hierarchy

    # Channels in TDMS are then mapped to the second level of the HDF5
    # hierarchy, under the appropriate groups.

    # Properties in TDMS are mapped to attributes in HDF5.
    # These all exist under the appropriate, channel group etc.

    h5file = h5py.File(filepath, mode)

    if group in h5file:
        container_group = h5file[group]
    else:
        container_group = h5file.create_group(group)

    # First write the properties at the root level
    for property_name, property_value in tdms_file.properties.items():
        container_group.attrs[property_name] = _hdf_attr_value(property_value)

    # Now iterate through groups and channels,
    # writing the properties and creating data sets
    datasets = {}
    for group in tdms_file.groups():
        # Write the group's properties
        container_group.create_group(group.name)
        for prop_name, prop_value in group.properties.items():
            container_group[group.name].attrs[prop_name] = _hdf_attr_value(prop_value)

        # Write properties and data for each channel
        for channel in group.channels():
            channel_key = group.name + '/' + channel.name

            if channel.data_type is types.String:
                # Encode as variable length UTF-8 strings
                datasets[channel.path] = container_group.create_dataset(
                    channel_key, (len(channel),), dtype=h5py.string_dtype())
            elif channel.data_type is types.TimeStamp:
                # Timestamps are represented as fixed length ASCII strings
                # because HDF doesn't natively support timestamps
                datasets[channel.path] = container_group.create_dataset(
                    channel_key, (len(channel),), dtype='S27')
            else:
                datasets[channel.path] = container_group.create_dataset(
                    channel_key, (len(channel),), dtype=channel.dtype)

            for prop_name, prop_value in channel.properties.items():
                container_group[channel_key].attrs[prop_name] = _hdf_attr_value(prop_value)

    # Set data
    if tdms_file.data_read:
        for group in tdms_file.groups():
            for channel in group.channels():
                datasets[channel.path][...] = _hdf_array(channel, channel.data)
    else:
        # Data hasn't been read into memory, stream it from disk
        for chunk in tdms_file.data_chunks():
            for group in chunk.groups():
                for channel_chunk in group.channels():
                    channel = tdms_file[group.name][channel_chunk.name]
                    offset = channel_chunk.offset
                    end = offset + len(channel_chunk)
                    datasets[channel.path][offset:end] = _hdf_array(channel, channel_chunk[:])

    return h5file


def _hdf_array(channel, data):
    """ Convert data array into a format suitable for initialising HDF data
    """
    if channel.data_type is types.TimeStamp:
        string_data = np.datetime_as_string(data, unit='us', timezone='UTC')
        return [s.encode('ascii') for s in string_data]
    return data


def _hdf_attr_value(value):
    """ Convert a value into a format suitable for an HDF attribute
    """
    if isinstance(value, np.datetime64):
        return np.string_(np.datetime_as_string(value, unit='us', timezone='UTC'))
    return value
