npTDMS
======

.. image:: https://app.wercker.com/status/446c67339f7d484188a35abc64dd3f51/s/master
    :alt: wercker status
    :target: https://app.wercker.com/project/bykey/446c67339f7d484188a35abc64dd3f51

npTDMS is a cross-platform Python package for reading and writing TDMS files as produced by LabVIEW,
and is built on top of the `numpy <http://www.numpy.org/>`__ package.
Data read from a TDMS file is stored in numpy arrays,
and numpy arrays are also used when writing TDMS file.

Typical usage when reading a TDMS file might look like::

    from nptdms import TdmsFile

    tdms_file = TdmsFile("path_to_file.tdms")
    channel = tdms_file.object('Group', 'Channel1')
    data = channel.data
    time = channel.time_track()
    # do stuff with data

And to write a file::

    from nptdms import TdmsWriter, ChannelObject
    import numpy

    with TdmsWriter("path_to_file.tdms") as tdms_writer:
        data_array = numpy.linspace(0, 1, 10)
        channel = ChannelObject('Group', 'Channel1', data_array)
        tdms_writer.write_segment([channel])

For more information, see the `npTDMS documentation <http://nptdms.readthedocs.io>`__.

Installation
------------

npTDMS is available from the Python Package Index, so the easiest way to
install it is by running (as root)::

    pip install npTDMS

Alternatively, after downloading the source code you can extract it and
change into the new directory, then run::

    python setup.py install

Links
-----

Source code lives at https://github.com/adamreeve/npTDMS and any issues can be
reported at https://github.com/adamreeve/npTDMS/issues.
Documentation is available at http://nptdms.readthedocs.io.

What Currently Doesn't Work
---------------------------

This module doesn't support TDMS files with XML headers or with
extended floating point data.

Contributors/Thanks
-------------------

Thanks to Floris van Vugt who wrote the pyTDMS module,
which helped when writing this module.

Thanks to Tony Perkins, Ruben De Smet, Martin Hochwallner and Peter Duncan
for contributing support for converting to Pandas DataFrames.

Thanks to nmgeek and jshridha for implementing support for DAQmx raw data
files.
