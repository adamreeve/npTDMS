Installation and Quick Start
============================

npTDMS is available from the Python Package Index, so the easiest way to
install it is by running (as root)::

    pip install npTDMS

Or you can install npTDMS as a non-root user inside your home directory::

    pip install --user npTDMS

Alternatively, after downloading the source code you can extract it and
change into the new directory, then run::

    python setup.py install

Typical usage when reading a TDMS file might look like::

    from nptdms import TdmsFile

    tdms_file = TdmsFile("path_to_file.tdms")
    channel = tdms_file.object('Group', 'Channel1')
    data = channel.data
    # do stuff with data

And to write a TDMS file::

    from nptdms import TdmsWriter, ChannelObject
    import numpy

    with TdmsWriter("path_to_file.tdms") as tdms_writer:
        data_array = numpy.linspace(0, 1, 10)
        channel = ChannelObject('Group', 'Channel1', data_array)
        tdms_writer.write_segment([channel])
