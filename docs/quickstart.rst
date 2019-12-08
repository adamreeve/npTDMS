Installation and Quick Start
============================

npTDMS is available from the Python Package Index, so the easiest way to
install it is by running (as root)::

    pip install npTDMS

Or you can install npTDMS as a non-root user inside your home directory::

    pip install --user npTDMS

There are optional features available that require additional dependencies.
These are ``hdf`` for hdf export, ``pandas`` for pandas DataFrame export,
and ``thermocouple_scaling`` for reading files that use thermocouple scalings.
You can specify these extra features when installing npTDMS to also install the dependencies they require::

    pip install npTDMS[hdf,pandas,thermocouple_scaling]

Alternatively, after downloading the source code you can extract it and
change into the new directory, then run::

    python setup.py install

Typical usage when reading a TDMS file might look like::

    from nptdms import TdmsFile

    tdms_file = TdmsFile("path_to_file.tdms")
    for group in tdms_file.groups():
        for channel in tdms_file.group_channels(group):
            # Access dictionary of properties:
            properties = channel.properties
            # Access numpy array of data for channel:
            data = channel.data
            # do stuff with data and properties...

And to write a TDMS file::

    from nptdms import TdmsWriter, ChannelObject
    import numpy

    with TdmsWriter("path_to_file.tdms") as tdms_writer:
        data_array = numpy.linspace(0, 1, 10)
        channel = ChannelObject('Group', 'Channel1', data_array)
        tdms_writer.write_segment([channel])
