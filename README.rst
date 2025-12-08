npTDMS
======

.. image:: https://img.shields.io/pypi/v/npTDMS.svg
    :alt: PyPI Version
    :target: https://pypi.org/project/npTDMS/
.. image:: https://github.com/adamreeve/npTDMS/actions/workflows/ci-cd.yml/badge.svg?branch=master&event=push
    :alt: Build status
    :target: https://github.com/adamreeve/npTDMS/actions/workflows/ci-cd.yml?query=event%3Apush+branch%3Amaster
.. image:: https://readthedocs.org/projects/nptdms/badge/?version=latest
    :target: https://nptdms.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/adamreeve/npTDMS/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/adamreeve/npTDMS
    :alt: Code coverage


npTDMS is a cross-platform Python package for reading and writing TDMS files as produced by LabVIEW,
and is built on top of the `numpy <http://www.numpy.org/>`__ package.
Data is read from TDMS files as numpy arrays,
and npTDMS also allows writing numpy arrays to TDMS files.

TDMS files are structured in a hierarchy of groups and channels.
A TDMS file can contain multiple groups, which may each contain multiple channels.
A file, group and channel may all have properties associated with them,
but only channels have array data.

Typical usage when reading a TDMS file might look like::

    from nptdms import TdmsFile

    tdms_file = TdmsFile.read("path_to_file.tdms")
    group = tdms_file['group name']
    channel = group['channel name']
    channel_data = channel[:]
    channel_properties = channel.properties

The ``TdmsFile.read`` method reads all data into memory immediately.
When you are working with large TDMS files or don't need to read all channel data,
you can instead use ``TdmsFile.open``. This is more memory efficient but
accessing data can be slower::

    with TdmsFile.open("path_to_file.tdms") as tdms_file:
        group = tdms_file['group name']
        channel = group['channel name']
        channel_data = channel[:]

npTDMS also has rudimentary support for writing TDMS files.
Using npTDMS to write a TDMS file looks like::

    from nptdms import TdmsWriter, ChannelObject
    import numpy

    with TdmsWriter("path_to_file.tdms") as tdms_writer:
        data_array = numpy.linspace(0, 1, 10)
        channel = ChannelObject('group name', 'channel name', data_array)
        tdms_writer.write_segment([channel])

For more detailed documentation on reading and writing TDMS files,
see the `npTDMS documentation <http://nptdms.readthedocs.io>`__.

Installation
------------

npTDMS is available from the Python Package Index, so the easiest way to
install it is by running::

    pip install npTDMS

There are optional features available that require additional dependencies.
These are `hdf` for hdf export, `pandas` for pandas DataFrame export, and
`thermocouple_scaling` for using thermocouple scalings. You can specify these
extra features when installing npTDMS to also install the dependencies they
require::

    pip install npTDMS[hdf,pandas,thermocouple_scaling]

Alternatively, after downloading the source code you can extract it and
change into the new directory, then run::

    python pip install .

Links
-----

Source code lives at https://github.com/adamreeve/npTDMS and any issues can be
reported at https://github.com/adamreeve/npTDMS/issues.
Documentation is available at http://nptdms.readthedocs.io.

Limitations
-----------

This module doesn't support TDMS files with XML headers or with
extended precision floating point data.

Contributors/Thanks
-------------------

Thanks to Floris van Vugt who wrote the pyTDMS module,
which helped when writing this module.

Thanks to Tony Perkins, Ruben De Smet, Martin Hochwallner and Peter Duncan
for contributing support for converting to Pandas DataFrames.

Thanks to nmgeek and jshridha for implementing support for DAQmx raw data
files.
