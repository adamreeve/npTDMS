"""Test writing TDMS files"""

from io import BytesIO
import logging
import numpy as np
import unittest

from nptdms import tdms, writer


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)
        logging.getLogger(writer.__name__).setLevel(logging.DEBUG)

    def test_data_write(self):
        a_input = np.linspace(0.0, 1.0, 100)
        b_input = np.linspace(0.0, 100.0, 100)

        a_segment = writer.ChannelSegment("group", "a", a_input)
        b_segment = writer.ChannelSegment("group", "b", b_input)

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
