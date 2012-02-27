npTDMS
======

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

For more information, see the `npTDMS documentation <http://readthedocs.org/docs/nptdms>`__.

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
Documentation is available at http://readthedocs.org/docs/npTDMS.

What Currently Doesn't Work
---------------------------

This module doesn't support TDMS files with XML headers or files with
string or extended floating point data.

Contributors/Thanks
-------------------

Thanks to Floris van Vugt who wrote the pyTDMS module,
which helped when writing this module.
