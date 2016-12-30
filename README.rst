npTDMS
======

.. image:: https://app.wercker.com/status/446c67339f7d484188a35abc64dd3f51/s/master
    :alt: wercker status
    :target: https://app.wercker.com/project/bykey/446c67339f7d484188a35abc64dd3f51

Cross-platform module for reading TDMS files as produced by LabView.
Data is stored as a numpy array, and is loaded using numpy's fromfile routine
so is very fast.

Typical usage might look like::

    #!/usr/bin/env python

    from nptdms import TdmsFile
    tdms_file = TdmsFile("path_to_file.tdms")
    channel = tdms_file.object('Group', 'Channel1')
    data = channel.data
    time = channel.time_track()
    # do stuff with data

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
