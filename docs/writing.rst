Writing TDMS files
==================

npTDMS has rudimentary support for writing TDMS files.
The full set of optimisations supported by the TDMS file format for
speeding up the writing of files and minimising file size are not
implemented by npTDMS, but the basic functionality required to
write TDMS files is available.

To write a TDMS file, the :py:class:`~nptdms.TdmsWriter` class is used, which
should be used as a context manager.
The :py:meth:`~nptdms.TdmsWriter.__init__` method accepts the path to the file to create, or a file
that has already been opened in binary write mode::

    with TdmsWriter("my_file.tdms") as tdms_writer:
        # write data

The :py:meth:`~nptdms.TdmsWriter.write_segment` method is used to write
a segment of data to the TDMS file. Because the TDMS file format is designed
for streaming data applications, it supports writing data one segment at a time
as data becomes available.
If you don't require this functionality you can simple call ``write_segment`` once
with all of your data.

The :py:meth:`~nptdms.TdmsWriter.write_segment` method takes a list of objects, each of which must be an
instance of one of:

- :py:class:`nptdms.RootObject`. This is the TDMS root object, and there may only be one root object in a segment.
- :py:class:`nptdms.GroupObject`. This is used to group the channel objects.
- :py:class:`nptdms.ChannelObject`. An object that contains data.
- :py:class:`nptdms.TdmsGroup` or :py:class:`nptdms.TdmsChannel`.
  A TDMS object that was read from a TDMS file using :py:class:`nptdms.TdmsFile`.

Each of :py:class:`~nptdms.RootObject`, :py:class:`~nptdms.GroupObject` and :py:class:`~nptdms.ChannelObject`
may optionally have properties associated with them, which
are passed into the ``__init__`` method as a dictionary.
The data types supported as property values are:

- Integers
- Floating point values
- Strings
- datetime or numpy datetime64 objects
- Boolean values

For more control over the data type used to represent a property value, for example
to use an unsigned integer type, you can pass an instance of one of the data types
from the :py:mod:`nptdms.types` module.

A complete example of writing a TDMS file with various object types and properties
is given below::

    from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject

    root_object = RootObject(properties={
        "prop1": "foo",
        "prop2": 3,
    })
    group_object = GroupObject("group_1", properties={
        "prop1": 1.2345,
        "prop2": False,
    })
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    channel_object = ChannelObject("group_1", "channel_1", data, properties={})

    with TdmsWriter("my_file.tdms") as tdms_writer:
        # Write first segment
        tdms_writer.write_segment([
            root_object,
            group_object,
            channel_object])
        # Write another segment with more data for the same channel
        more_data = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        channel_object = ChannelObject("group_1", "channel_1", more_data, properties={})
        tdms_writer.write_segment([channel_object])

You could also read a TDMS file and then re-write it by passing
:py:class:`~nptdms.TdmsGroup` and :py:class:`~nptdms.TdmsChannel`
instances to the ``write_segment`` method. If you want
to only copy certain channels for example, you could do something like::

    from nptdms import TdmsFile, TdmsWriter, RootObject

    original_file = TdmsFile("original_file.tdms")
    original_groups = original_file.groups()
    original_channels = [chan for group in original_groups for chan in group.channels()]

    with TdmsWriter("copied_file.tdms") as copied_file:
        root_object = RootObject(original_file.properties)
        channels_to_copy = [chan for chan in original_channels if include_channel(chan)]
        copied_file.write_segment([root_object] + original_groups + channels_to_copy)

Note that this isn't suitable for copying channels with scaled data, as the channel data
will already have scaling applied.
