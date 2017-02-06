Reading TDMS files
==================

To read a TDMS file, create an instance of the :py:class:`nptdms.TdmsFile`
class, passing the path to the file, or an already opened file to the ``__init__`` method::

    tdms_file = TdmsFile("my_file.tdms")

This will read the contents of the TDMS file, then the various objects
in the file can be accessed using the
:py:meth:`nptdms.TdmsFile.object` method.
An object in a TDMS file is either the root object, a group object, or a channel
object.
Only channel objects contain data, but any object may have properties associated with it.
For example, it is common to have properties stored against the root object such as the
file title and author.

The object returned by the ``object`` method is an instance of :py:class:`nptdms.TdmsObject`.
If this is a channel containing data, you can access the data as a numpy array using its
``data`` attribute::

    channel_object = tdms_file.object("group_name", "channel_name")
    data = channel_object.data

If the array is waveform data and has the ``wf_start_offset`` and ``wf_increment``
properties, you can get an array of relative time values for the data using the
:py:meth:`nptdms.TdmsObject.time_track` method::

    time = channel_object.time_track()

You can access the properties of an object using the :py:meth:`nptdms.TdmsObject.property` method,
or the :py:attr:`nptdms.TdmsObject.properties` dictionary, for example::

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
