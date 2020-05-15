npTDMS API Reference
====================

Reading TDMS Files
------------------

.. module:: nptdms

.. autoclass:: TdmsFile()
  :members:
  :exclude-members: object, objects, group_channels, channel_data

.. autoclass:: TdmsGroup()
  :members:
  :exclude-members: group, channel, has_data, property

.. autoclass:: TdmsChannel()
  :members:
  :exclude-members: group, channel, has_data, property, number_values

.. autoclass:: DataChunk()
  :members:

.. autoclass:: GroupDataChunk()
  :members:

.. autoclass:: ChannelDataChunk()
  :members:

Writing TDMS Files
------------------

.. autoclass:: TdmsWriter
  :members:

  .. automethod:: __init__

.. autoclass:: RootObject
  :members:

  .. automethod:: __init__

.. autoclass:: GroupObject
  :members:

  .. automethod:: __init__

.. autoclass:: ChannelObject
  :members:

  .. automethod:: __init__

Data Types for Property Values
------------------------------

.. module:: nptdms.types

.. autoclass:: Int8

.. autoclass:: Int16

.. autoclass:: Int32

.. autoclass:: Int64

.. autoclass:: Uint8

.. autoclass:: Uint16

.. autoclass:: Uint32

.. autoclass:: Uint64

.. autoclass:: SingleFloat

.. autoclass:: DoubleFloat

.. autoclass:: String

.. autoclass:: Boolean

.. autoclass:: TimeStamp

Timestamps
----------

.. module:: nptdms.timestamp

.. autoclass:: TdmsTimestamp
  :members:

.. autoclass:: TimestampArray()
  :members:

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
