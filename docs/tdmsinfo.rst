The tdmsinfo Command
====================

npTDMS comes with a command line program, ``tdmsinfo``, which
lists the contents of a TDMS file.
Usage looks like::

    tdmsinfo [--properties] tdms_file

Passing the ``--properties`` or ``-p`` argument will include TDMS object
properties in the printed information as well as the data type and length of channels.

The output of tdmsinfo including properties will look something like::

    /
      properties:
        name: test_file
      /'group_1'
        properties:
          group_property: property value
        /'group_1'/'channel_1'
          data type: Uint32
          length: 2000
          properties:
            wf_start_time: 2016-12-3014:56:00+00:00
            wf_increment: 0.0005
            wf_samples: 200

There is also a ``--debug`` or ``-d`` argument that will output debug information
to stderr, which can be useful when debugging a problem with a TDMS file.
