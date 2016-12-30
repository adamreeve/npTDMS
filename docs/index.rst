Welcome to npTDMS's documentation
=================================

npTDMS is a Python module for reading binary TDMS files produced by LabView,
and is based on the
`file format documentation <http://zone.ni.com/devzone/cda/tut/p/id/5696>`_
released by NI.

Data is read from file directly into memory and is then interpreted as a
NumPy array, so reading files should be very fast.

Installation
============

npTDMS is available from the Python Package Index, so the easiest way to
install it is by running (as root)::

    pip install npTDMS

Alternatively, after downloading the source code you can extract it and
change into the new directory, then run::

    python setup.py install

Quick Start
===========

Typical usage might look like::

    #!/usr/bin/env python

    from nptdms import TdmsFile
    tdms_file = TdmsFile("path_to_file.tdms")
    channel = tdms_file.object('Group', 'Channel1')
    data = channel.data
    time = channel.time_track()
    # do stuff with data

tdmsinfo Program
================

npTDMS comes with a command line program, ``tdmsinfo``, which
lists the contents of a TDMS file.
Usage looks like::

    tdmsinfo [--properties] tdms_file

Passing the ``--properties`` or ``-p`` argument will include TDMS object
properties in the printed information.

What Doesn't Work
=================

Reading TDMS files with XML headers or files with
extended floating point data currently does not work.

Reference
=========

.. module:: nptdms.tdms

.. autoclass:: TdmsFile
  :members:

  .. automethod:: __init__

.. autoclass:: TdmsObject
  :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
