""" Python module for reading TDMS files produced by LabView

    This module contains the public facing API
"""

from nptdms.utils import Timer, OrderedDict
from nptdms.log import log_manager
from nptdms.common import path_components
from nptdms.reader import TdmsReader
from nptdms.data_store import get_data_store


log = log_manager.get_logger(__name__)


class TdmsFile(object):
    """Reads and stores data from a TDMS file.
    """

    def __init__(self, file, memmap_dir=None, read_metadata_only=False):
        """Initialise a new TDMS file object, reading all data.

        :param file: Either the path to the tdms file to read or an already
            opened file.
        :param memmap_dir: The directory to store memmapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        :param read_metadata_only: If this parameter is enabled then the
            metadata of the TDMS file will only be read.
        """

        self._memmap_dir = memmap_dir
        self._obj_data = {}

        if hasattr(file, "read"):
            # Is a file
            self._read_file(file, memmap_dir, read_metadata_only)
        else:
            # Is path to a file
            with open(file, 'rb') as open_file:
                self._read_file(open_file, memmap_dir, read_metadata_only)


    def _read_file(self, tdms_file, memmap_dir, read_metadata_only):
        reader = TdmsReader(tdms_file)
        reader.read_metadata()
        self._objects = reader.objects

        if not read_metadata_only:
            self._read_data(reader)

    def _read_data(self, tdms_reader):
        with Timer(log, "Allocate space"):
            # Allocate space for data
            for obj in self._objects.values():
                self._obj_data = get_data_store(obj)

        with Timer(log, "Read data"):
            # Now actually read all the data
            for segment in tdms_reader.read_data():
                for (path, data) in segment.raw_data.items():
                    data_store = self._obj_data[path]
                    data_store.append_data(data)
                for (path, data) in segment.daqmx_raw_data:
                    data_store = self._obj_data[path]
                    for scaler_id, scaler_data in data.items():
                        data_store.append_data(scaler_id, scaler_data)

    def object(self, *path):
        """Get a TDMS object from the file

        :param path: The object group and channel. Providing no channel
            returns a group object, and providing no channel or group
            will return the root object.
        :rtype: :class:`TdmsObject`

        For example, to get the root object::

            object()

        To get a group::

            object("group_name")

        To get a channel::

            object("group_name", "channel_name")
        """

        object_path = _components_to_path(*path)
        try:
            return self._objects[object_path]
        except KeyError:
            raise KeyError("Invalid object path: %s" % object_path)

    def groups(self):
        """Return the names of groups in the file

        Note that there is not necessarily a TDMS object associated with
        each group name.

        :rtype: List of strings.

        """

        # Split paths into components and take the first (group) component.
        object_paths = (
            path_components(path)
            for path in self._objects)
        group_names = (path[0] for path in object_paths if len(path) > 0)

        # Use an ordered dict as an ordered set to find unique
        # groups in order.
        groups_set = OrderedDict()
        for group in group_names:
            groups_set[group] = None
        return list(groups_set)

    def group_channels(self, group):
        """Returns a list of channel objects for the given group

        :param group: Name of the group to get channels for.
        :rtype: List of :class:`TdmsObject` objects.

        """

        path = _components_to_path(group)
        return [
            self._objects[p]
            for p in self._objects
            if p.startswith(path + '/')]

    def channel_data(self, group, channel):
        """Get the data for a channel

        :param group: The name of the group the channel is in.
        :param channel: The name of the channel to get data for.
        :returns: The channel data.
        :rtype: NumPy array.

        """

        return self.object(group, channel).data

    def properties(self):
        """ Return the properties of this file

        These are properties associated with the root object of the file.
        """

        try:
            obj = self.object()
            return obj.properties
        except KeyError:
            return {}

    def as_dataframe(self, time_index=False, absolute_time=False):
        """
        Converts the TDMS file to a DataFrame

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        dataframe_dict = OrderedDict()
        for key, value in self._objects.items():
            if value.has_data:
                index = value.time_track(absolute_time) if time_index else None
                dataframe_dict[key] = pd.Series(data=value.data, index=index)
        return pd.DataFrame.from_dict(dataframe_dict)

    def as_hdf(self, filepath, mode='w', group='/'):
        """
        Converts the TDMS file into an HDF5 file

        :param filepath: The path of the HDF5 file you want to write to.
        :param mode: The write mode of the HDF5 file. This can be w, a ...
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
            root = self.object()
            for property_name, property_value in root.properties.items():
                container_group.attrs[property_name] = property_value
        except KeyError:
            # No root object present
            pass

        # Now iterate through groups and channels,
        # writing the properties and data
        for group_name in self.groups():
            try:
                group = self.object(group_name)

                # Write the group's properties
                for prop_name, prop_value in group.properties.items():
                    container_group[group_name].attrs[prop_name] = prop_value

            except KeyError:
                # No group object present
                pass

            # Write properties and data for each channel
            for channel in self.group_channels(group_name):
                for prop_name, prop_value in channel.properties.items():
                    container_group.attrs[prop_name] = prop_value

                container_group[group_name+'/'+channel.channel] = channel.data

        return h5file


class Group(object):
    """A group of channels in a TDMS file

    A group has no associated data but may have properties
    """

    def __init__(self, name, channels):
        self._name = name
        self._channels = OrderedDict((c.name, c) for c in channels)

    @property
    def name(self):
        """The name of this group
        """
        return self._name

    def channels(self):
        """Returns a list of channel objects within this group
        """
        return list(self._channels.values())

    def channel(self, channel_name):
        """Get a channel from this group by name
        """
        try:
            return self._channels[channel_name]
        except KeyError:
            raise ValueError(
                "Group %s has no channel %s" % self._name, channel_name)

    def __str__(self):
        return self._name

    def __repr__(self):
        return "<Group name='%s'>" % self._name


class Channel(object):
    """A channel from a TDMS file which may contain properties and data
    """

    def __init__(self):
        pass


def _components_to_path(*args):
    """Convert group and channel to object path"""

    return ('/' + '/'.join(
        ["'" + arg.replace("'", "''") + "'" for arg in args]))
