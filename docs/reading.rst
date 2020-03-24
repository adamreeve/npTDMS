Reading TDMS files
==================

To read a TDMS file, create an instance of the :py:class:`~nptdms.TdmsFile`
class using the static read method, passing the path to the file, or an already opened file::

    tdms_file = TdmsFile.read("my_file.tdms")

This will read all of the contents of the TDMS file, then groups within the file
can be accessed using the
:py:meth:`~nptdms.TdmsFile.groups` method, or by indexing into the file with a group name::

    all_groups = tdms_file.groups()
    group = tdms_file["group name"]

A group is an instance of the :py:class:`~nptdms.TdmsGroup` class,
and can contain multiple channels of data. You can access channels in a group with the
:py:meth:`~nptdms.TdmsGroup.channels` method or by indexing into the group with a channel name::

    all_group_channels = group.channels()
    channel = group["channel name"]

Channels are instances of the :py:class:`~nptdms.TdmsChannel` class
and have a ``data`` attribute for accessing the channel data as a numpy array::

    data = channel.data

If the array is waveform data and has the ``wf_start_offset`` and ``wf_increment``
properties, you can get an array of relative time values for the data using the
:py:meth:`~nptdms.TdmsChannel.time_track` method::

    time = channel.time_track()

In addition, if the ``wf_start_time`` property is set,
you can pass ``absolute_time=True`` to get an array of absolute times in UTC.

A TDMS file, group and channel can all have properties associated with them, so each of the
:py:class:`~nptdms.TdmsFile`, :py:class:`~nptdms.TdmsGroup` and :py:class:`~nptdms.TdmsChannel`
classes provide access to these properties as a dictionary using their ``properties`` property::

    # Iterate over all items in the file properties and print them
    for name, value in tdms_file.properties.items():
        print("{0}: {1}".format(name, value))

    # Get a single property value
    property_value = tdms_file.property("my_property_name")

    # Get a group property
    property_value = tdms_file["group name"].properties["group_property_name"]

    # Get a channel property
    property_value = tdms_file["group name"]["channel name"].properties["channel_property_name"]

Timestamps
----------

Timestamps are represented by numpy datetime64 objects with microsecond precision.
Note that TDMS files are capable of storing times with a precision of 2 :sup:`-64` seconds,
so some precision is lost when reading them in npTDMS.

Timestamps in TDMS files are stored in UTC time and npTDMS does not do any timezone conversions.
If you would like to convert a time from a TDMS file to your local timezone,
the arrow package is recommended. For example::

    import datetime
    import arrow

    timestamp = channel.properties['wf_start_time']
    local_time = arrow.get(timestamp.astype(datetime.datetime)).to('local')
    print(local_time.format())

Here we first convert the numpy datetime64 object to Python's built in datetime type before converting it to an arrow time,
then convert it from UTC to the local timezone.

Scaled data
-----------

The TDMS format supports different ways of scaling data, and DAQmx raw data in particular is usually scaled.
The :py:attr:`~nptdms.TdmsChannel.data` property of the channel returns this scaled data.
You can additionally use the :py:attr:`~nptdms.TdmsChannel.raw_data` property to access the unscaled data.
Note that DAQmx channels may have multiple raw scalers rather than a single raw data channel,
in which case you need to use the :py:attr:`~nptdms.TdmsChannel.raw_scaler_data`
property to access the raw data as a dictionary of scaler id to raw data array.

Conversion to other formats
---------------------------

npTDMS has convenience methods to convert data to Pandas DataFrames or HDF5 files.
The :py:class:`~nptdms.TdmsFile` class has :py:meth:`~nptdms.TdmsFile.as_dataframe` and
:py:meth:`~nptdms.TdmsFile.as_hdf` methods to convert a whole file to a DataFrame or HDF5 file.
In addition there is an :py:meth:`~nptdms.TdmsGroup.as_dataframe` method on :py:class:`~nptdms.TdmsGroup`
and an :py:meth:`~nptdms.TdmsGroup.as_dataframe` method on :py:class:`~nptdms.TdmsChannel`
for converting a single group or channel to a Pandas DataFrame.
