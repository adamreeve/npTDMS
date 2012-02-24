======
npTDMS
======

Cross-platform module for reading TDMS files as produced by LabView, based
on the file format description at http://zone.ni.com/devzone/cda/tut/p/id/5696.

Data is returned as a numpy array, and if the data storage is contiguous then
numpy's fromfile is used so that loading is very fast.

Typical usage might look like::

    #!/usr/bin/env python

    from nptdms import tdms
    tdms_file = tdms.TdmsFile("path_to_file.tdms")
    object = tdms_file.object('Group', 'Channel1')
    data = object.data
    time = object.time_track()
    # do stuff with data

What Currently Doesn't Work
---------------------------

This module doesn't support TDMS files with XML headers and some data types,
eg. strings, time stamps and extended floating point values.

Contributors/Thanks
-------------------

Thanks to Floris van Vugt who wrote the pyTDMS module,
which helped when writing this module.
