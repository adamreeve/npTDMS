"""Test writing TDMS files"""

from datetime import datetime
from io import BytesIO
import logging
import numpy as np
import os
import tempfile
import unittest
try:
    import pytz
except ImportError:
    pytz = None

from nptdms import tdms, writer
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)
        logging.getLogger(writer.__name__).setLevel(logging.DEBUG)

    def test_can_read_tdms_file_after_writing(self):
        a_input = np.linspace(0.0, 1.0, 100)
        b_input = np.linspace(0.0, 100.0, 100)

        a_segment = ChannelObject("group", "a", a_input)
        b_segment = ChannelObject("group", "b", b_input)

        output_file = BytesIO()
        with TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([a_segment, b_segment])

        output_file.seek(0)
        tdms_file = TdmsFile(output_file)

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

        a_segment = RootObject(properties={
            "prop1": "foo",
            "prop2": 3,
        })
        b_segment = GroupObject("group_name", properties={
            "prop3": 1.2345,
            "prop4": test_time,
        })

        output_file = BytesIO()
        with TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([a_segment, b_segment])

        output_file.seek(0)
        tdms_file = TdmsFile(output_file)

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

    def test_can_write_multiple_segments(self):
        input_1 = np.linspace(0.0, 1.0, 10)
        input_2 = np.linspace(2.0, 3.0, 10)

        segment_1 = ChannelObject("group", "a", input_1)
        segment_2 = ChannelObject("group", "a", input_2)

        output_file = BytesIO()
        with TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([segment_1])
            tdms_writer.write_segment([segment_2])

        output_file.seek(0)
        tdms_file = TdmsFile(output_file)

        output_data = tdms_file.object("group", "a").data

        expected_data = np.append(input_1, input_2)
        self.assertEqual(len(output_data), len(expected_data))
        self.assertTrue((output_data == expected_data).all())

    def test_can_write_to_file_using_path(self):
        input_1 = np.linspace(0.0, 1.0, 10)
        segment = ChannelObject("group", "a", input_1)

        tempdir = tempfile.mkdtemp()
        temppath = "%s/test_file.tdms" % tempdir
        try:
            with TdmsWriter(temppath) as tdms_writer:
                tdms_writer.write_segment([segment])
        finally:
            if os.path.exists(temppath):
                os.remove(temppath)
            os.rmdir(tempdir)

    def test_can_append_to_file_using_path(self):
        input_1 = np.linspace(0.0, 1.0, 10)
        input_2 = np.linspace(1.0, 2.0, 10)
        segment_1 = ChannelObject("group", "a", input_1)
        segment_2 = ChannelObject("group", "a", input_2)

        tempdir = tempfile.mkdtemp()
        temppath = "%s/test_file.tdms" % tempdir
        try:
            with TdmsWriter(temppath) as tdms_writer:
                tdms_writer.write_segment([segment_1])
            with TdmsWriter(temppath, 'a') as tdms_writer:
                tdms_writer.write_segment([segment_2])

            tdms_file = TdmsFile(temppath)

            output = tdms_file.object("group", "a").data

            self.assertEqual(len(output), 20)
            np.testing.assert_almost_equal(
                output, np.concatenate([input_1, input_2]))

        finally:
            if os.path.exists(temppath):
                os.remove(temppath)
            os.rmdir(tempdir)

    def test_can_write_to_file_using_open_file(self):
        input_1 = np.linspace(0.0, 1.0, 10)
        segment = ChannelObject("group", "a", input_1)

        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            with TdmsWriter(output_file.file) as tdms_writer:
                tdms_writer.write_segment([segment])
