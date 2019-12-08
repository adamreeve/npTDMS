Reading TDMS files
==================

To read a TDMS file, create an instance of the :py:class:`~nptdms.TdmsFile`
class, passing the path to the file, or an already opened file to the ``__init__`` method::

    tdms_file = nptdms.TdmsFile("my_file.tdms")

This will read the contents of the TDMS file, then the various objects
in the file can be accessed using the
:py:meth:`~nptdms.TdmsFile.object` method.
An object in a TDMS file is either the root object, a group object, or a channel
object.
Only channel objects contain data, but any object may have properties associated with it.
For example, it is common to have properties stored against the root object such as the
file title and author.

If you don't already know what groups and channels are present in your file,
you can use the :py:meth:`~nptdms.TdmsFile.groups` method to get all of the groups
in a file, and then the :py:meth:`~nptdms.TdmsFile.group_channels` method to get all
of the channels in a group.

The object returned by the :py:meth:`~nptdms.TdmsFile.object` method
is an instance of :py:class:`~nptdms.TdmsObject`.
If this is a channel containing data, you can access the data as a numpy array using its
``data`` attribute::

    channel_object = tdms_file.object("group_name", "channel_name")
    data = channel_object.data

If the array is waveform data and has the ``wf_start_offset`` and ``wf_increment``
properties, you can get an array of relative time values for the data using the
:py:meth:`~nptdms.TdmsObject.time_track` method::

    time = channel_object.time_track()

In addition, if the ``wf_start_time`` property is set,
you can pass ``absolute_time=True`` to get an array of absolute times in UTC.

You can access the properties of an object using the :py:meth:`~nptdms.TdmsObject.property` method,
or the ``properties`` dictionary, for example::

    root_object = tdms_file.object()

    # Iterate over all items in the properties dictionary and print them
    for name, value in root_object.properties.items():
        print("{0}: {1}".format(name, value))

    # Get a single property value
    property_value = root_object.property("my_property_name")

You may also have group objects in your TDMS files that do not contain
channels but are only used to group properties, in which case you can access
these objects in the same way as a normal group::

    attributes = tdms_file.object("attributes").properties

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

    timestamp = tdms_object.properties['wf_start_time']
    local_time = arrow.get(timestamp.astype(datetime.datetime)).to('local')
    print(local_time.format())

Here we first convert the numpy datetime64 object to Python's built in datetime type before converting it to an arrow time,
then convert it from UTC to the local timezone.

Scaled data
-----------

The TDMS format supports different ways of scaling data, and DAQmx raw data in particular is usually scaled.
The :py:attr:`~nptdms.TdmsObject.data` attribute of the channel returns this scaled data.
You can additionally use the :py:attr:`~nptdms.TdmsObject.raw_data` attribute to access the unscaled data.
Note that DAQmx channels may have multiple raw scalers rather than a single raw data channel,
in which case you need to use the :py:meth:`~nptdms.TdmsObject.raw_scaler_data`
method to access the raw data for a specific scaler id.

Conversion to other formats
---------------------------

npTDMS has convenience methods to convert data to Pandas DataFrames or HDF5 files.
The :py:class:`~nptdms.TdmsFile` class has :py:meth:`~nptdms.TdmsFile.as_dataframe` and
:py:meth:`~nptdms.TdmsFile.as_hdf` methods to convert a whole file to a DataFrame or HDF5 file.
In addition, the :py:class:`~nptdms.TdmsObject` has a :py:meth:`~nptdms.TdmsObject.as_dataframe` method
for converting a single group or channel to a Pandas DataFrame.
