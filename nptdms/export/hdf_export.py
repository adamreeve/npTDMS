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

    container_group = None
    if group in h5file:
        container_group = h5file[group]
    else:
        container_group = h5file.create_group(group)

    # First write the properties at the root level
    try:
        root = tdms_file.object()
        for property_name, property_value in root.properties.items():
            container_group.attrs[property_name] = _hdf_attr_value(property_value)
    except KeyError:
        # No root object present
        pass

    # Now iterate through groups and channels,
    # writing the properties and data
    for group_name in tdms_file.groups():
        try:
            group = tdms_file.object(group_name)

            # Write the group's properties
            container_group.create_group(group_name)
            for prop_name, prop_value in group.properties.items():
                container_group[group_name].attrs[prop_name] = _hdf_attr_value(prop_value)

        except KeyError:
            # No group object present
            pass

        # Write properties and data for each channel
        for channel in tdms_file.group_channels(group_name):
            channel_key = group_name + '/' + channel.channel

            if channel.data_type is types.String:
                # Encode as variable length UTF-8 strings
                channel_data = container_group.create_dataset(
                    channel_key, (len(channel.data),), dtype=h5py.string_dtype())
                channel_data[...] = channel.data
            elif channel.data_type is types.TimeStamp:
                # Timestamps are represented as fixed length ASCII strings
                # because HDF doesn't natively support timestamps
                channel_data = container_group.create_dataset(
                    channel_key, (len(channel.data),), dtype='S27')
                string_data = np.datetime_as_string(channel.data, unit='us', timezone='UTC')
                encoded_data = [s.encode('ascii') for s in string_data]
                channel_data[...] = encoded_data
            else:
                container_group[channel_key] = channel.data

            for prop_name, prop_value in channel.properties.items():
                container_group[channel_key].attrs[prop_name] = _hdf_attr_value(prop_value)

    return h5file


def _hdf_attr_value(value):
    """ Convert a value into a format suitable for an HDF attribute
    """
    if isinstance(value, np.datetime64):
        return np.string_(np.datetime_as_string(value, unit='us', timezone='UTC'))
    return value
