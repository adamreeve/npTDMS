Reading TDMS files
==================

To read a TDMS file, create an instance of the :py:class:`~nptdms.TdmsFile`
class using one of the static :py:meth:`nptdms.TdmsFile.read` or :py:meth:`nptdms.TdmsFile.open` methods,
passing the path to the file, or an already opened file.
The :py:meth:`~nptdms.TdmsFile.read` method will read all channel data immediately::

    tdms_file = TdmsFile.read("my_file.tdms")

If using the :py:meth:`~nptdms.TdmsFile.open` method, only the file metadata will be read initially,
and the returned :py:class:`~nptdms.TdmsFile` object should be used as a context manager to keep
the file open and allow channel data to be read on demand::

    with TdmsFile.open("my_file.tdms") as tdms_file:
        # Use tdms_file
        ...

Using an instance of :py:class:`~nptdms.TdmsFile`, groups within the file
can be accessed by indexing into the file with a group name, or all groups
can be retrieved as a list with the :py:meth:`~nptdms.TdmsFile.groups` method::

    group = tdms_file["group name"]
    all_groups = tdms_file.groups()

A group is an instance of the :py:class:`~nptdms.TdmsGroup` class,
and can contain multiple channels of data.
You can access channels in a group by indexing into the group with a channel name
or retrieve all channels as a list with the :py:meth:`~nptdms.TdmsGroup.channels` method::

    channel = group["channel name"]
    all_group_channels = group.channels()

Channels are instances of the :py:class:`~nptdms.TdmsChannel` class
and act like arrays. They can be indexed with an integer index to retrieve
a single value or with a slice to retrieve all data or a subset of data
as a numpy array::

    all_channel_data = channel[:]
    data_subset = channel[100:200]
    first_channel_value = channel[0]

If the channel contains waveform data and has the ``wf_start_offset`` and ``wf_increment``
properties, you can get an array of relative time values for the data using the
:py:meth:`~nptdms.TdmsChannel.time_track` method::

    time = channel.time_track()

In addition, if the ``wf_start_time`` property is set,
you can pass ``absolute_time=True`` to get an array of absolute times in UTC.

A TDMS file, group and channel can all have properties associated with them, so each of the
:py:class:`~nptdms.TdmsFile`, :py:class:`~nptdms.TdmsGroup` and :py:class:`~nptdms.TdmsChannel`
classes provide access to these properties as a dictionary using their ``properties`` attribute::

    # Iterate over all items in the file properties and print them
    for name, value in tdms_file.properties.items():
        print("{0}: {1}".format(name, value))

    # Get a single property value from the file
    property_value = tdms_file.properties["my_property_name"]

    # Get a group property
    property_value = tdms_file["group name"].properties["group_property_name"]

    # Get a channel property
    property_value = tdms_file["group name"]["channel name"].properties["channel_property_name"]

In addition to the properties dictionary,
all groups and channels have ``name`` and ``path`` attributes.
The name is the human readable name of the group or channel, and the path
is the full path to the TDMS object, which includes the group name for channels::

    group = tdms_file["group name"]
    channel = group["channel name"]
    print(group.name)    # Prints "group name"
    print(group.path)    # Prints "/'group name'"
    print(channel.name)  # Prints "channel name"
    print(channel.path)  # Prints "/'group name'/'channel name'"

Reading large files
-------------------

TDMS files are often too large to easily fit in memory so npTDMS offers a few ways to deal with this.
A TDMS file can be opened for reading without reading all the data immediately
using the static :py:meth:`~nptdms.TdmsFile.open` method,
then channel data is read as required::

    with TdmsFile.open(tdms_file_path) as tdms_file:
        channel = tdms_file[group_name][channel_name]
        all_channel_data = channel[:]
        data_subset = channel[100:200]

TDMS files are written in multiple segments, where each segment can in turn have
multiple chunks of data.
When accessing a value or a slice of data in a channel, npTDMS will read whole chunks at a time.
npTDMS also allows streaming data from a file chunk by chunk using
:py:meth:`nptdms.TdmsFile.data_chunks`. This is a generator that produces instances of
:py:class:`~nptdms.DataChunk`.
For example, to compute the mean of a channel::

    channel_sum = 0.0
    channel_length = 0
    with TdmsFile.open(tdms_file_path) as tdms_file:
        for chunk in tdms_file.data_chunks():
            channel_chunk = chunk[group_name][channel_name]
            channel_length += len(channel_chunk)
            channel_sum += channel_chunk[:].sum()
    channel_mean = channel_sum / channel_length

This approach can be useful to stream TDMS data to another format on disk or into a data store.
It's also possible to stream data chunks for a single channel using :py:meth:`nptdms.TdmsChannel.data_chunks`::

    with TdmsFile.open(tdms_file_path) as tdms_file:
        channel = tdms_file[group_name][channel_name]
        for chunk in channel.data_chunks():
            channel_chunk_data = chunk[:]

If you don't need to read the channel data at all and only need to read metadata, you can
also use the static :py:meth:`~nptdms.TdmsFile.read_metadata` method::

    tdms_file = TdmsFile.read_metadata(tdms_file_path)

In cases where you need to work with large arrays of channel data as if all data was in memory,
you can also pass the ``memmap_dir`` argument when reading a file.
This will read data into memory mapped numpy arrays on disk,
and your operating system will then page data in and out of memory as required::

    with tempfile.TemporaryDirectory() as temp_memmap_dir:
        tdms_file = TdmsFile.read(tdms_file_path, memmap_dir=temp_memmap_dir)

Timestamps
----------

Timestamps are represented by numpy datetime64 objects with microsecond precision.
Note that TDMS files are capable of storing times with a precision of 2 :sup:`-64` seconds,
so some precision is lost when reading them in npTDMS.

Timestamps in TDMS files are stored in UTC time and npTDMS does not do any timezone conversions.
If timestamps need to be converted to the local timezone,
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
The data retrieved from a :py:attr:`~nptdms.TdmsChannel` has scaling applied.
If you have opened a TDMS file with :py:meth:`~nptdms.TdmsFile.read`,
you can access the raw unscaled data with the :py:attr:`~nptdms.TdmsChannel.raw_data` property of a channel.
Note that DAQmx channels may have multiple raw scalers rather than a single raw data channel,
in which case you need to use the :py:attr:`~nptdms.TdmsChannel.raw_scaler_data`
property to access the raw data as a dictionary of scaler id to raw data array.

When you've opened a TDMS file with :py:meth:`~nptdms.TdmsFile.open`, you instead need to use
:py:attr:`~nptdms.TdmsChannel.read_data`, passing ``scaled=False``::

    with TdmsFile.open(tdms_file_path) as tdms_file:
        channel = tdms_file[group_name][channel_name]
        unscaled_data = channel.read_data(scaled=False)

This will return an array of raw data, or a dictionary of scaler id to raw scaler data for DAQmx data.

Conversion to other formats
---------------------------

npTDMS has convenience methods to convert data to Pandas DataFrames or HDF5 files.
The :py:class:`~nptdms.TdmsFile` class has :py:meth:`~nptdms.TdmsFile.as_dataframe` and
:py:meth:`~nptdms.TdmsFile.as_hdf` methods to convert a whole file to a DataFrame or HDF5 file.
In addition there is an :py:meth:`~nptdms.TdmsGroup.as_dataframe` method on :py:class:`~nptdms.TdmsGroup`
and an :py:meth:`~nptdms.TdmsGroup.as_dataframe` method on :py:class:`~nptdms.TdmsChannel`
for converting a single group or channel to a Pandas DataFrame.
