"""Test writing TDMS files"""

from datetime import datetime
from io import BytesIO
import logging
import numpy as np
import unittest
try:
    import pytz
except ImportError:
    pytz = None

from nptdms import tdms, writer


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)
        logging.getLogger(writer.__name__).setLevel(logging.DEBUG)

    def test_can_read_tdms_file_after_writing(self):
        a_input = np.linspace(0.0, 1.0, 100)
        b_input = np.linspace(0.0, 100.0, 100)

        a_segment = writer.ChannelObject("group", "a", a_input)
        b_segment = writer.ChannelObject("group", "b", b_input)

        output_file = BytesIO()
        with writer.TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([a_segment, b_segment])

        output_file.seek(0)
        tdms_file = tdms.TdmsFile(output_file)

        a_output = tdms_file.object("group", "a").data
        b_output = tdms_file.object("group", "b").data

        self.assertEqual(len(a_output), len(a_input))
        self.assertEqual(len(b_output), len(b_input))
        self.assertTrue((a_output == a_input).all())
        self.assertTrue((b_output == b_input).all())

    def test_can_read_tdms_file_properties_after_writing(self):
        test_time = datetime.utcnow()
        if pytz:
            test_time = test_time.replace(tzinfo=pytz.utc)

        a_segment = writer.RootObject(properties={
            "prop1": "foo",
            "prop2": 3,
        })
        b_segment = writer.GroupObject("group_name", properties={
            "prop3": 1.2345,
            "prop4": test_time,
        })

        output_file = BytesIO()
        with writer.TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([a_segment, b_segment])

        output_file.seek(0)
        tdms_file = tdms.TdmsFile(output_file)

        a_output = tdms_file.object()
        b_output = tdms_file.object("group_name")

        self.assertTrue("prop1" in a_output.properties, msg="prop1 not found")
        self.assertTrue("prop2" in a_output.properties, msg="prop2 not found")
        self.assertTrue("prop3" in b_output.properties, msg="prop3 not found")
        self.assertTrue("prop4" in b_output.properties, msg="prop4 not found")
        self.assertEqual(a_output.properties["prop1"], "foo")
        self.assertEqual(a_output.properties["prop2"], 3)
        self.assertEqual(b_output.properties["prop3"], 1.2345)
        self.assertEqual(b_output.properties["prop4"], test_time)
