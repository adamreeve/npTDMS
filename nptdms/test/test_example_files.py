""" Test reading example TDMS files
"""

import os
import numpy as np
from nptdms import tdms


DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/data'


def test_labview_file():
    """Test reading a file that was created by LabVIEW"""
    test_file = tdms.TdmsFile(DATA_DIR + '/Digital_Input.tdms')
    group = ("07/09/2012 06:58:23 PM - " +
             "Digital Input - Decimated Data_Level1")
    channel = "Dev1_port3_line7 - line 0"
    expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    data = test_file.object(group, channel).data
    np.testing.assert_almost_equal(data[:10], expected)


def test_raw_format():
    """Test reading a file with DAQmx raw data"""
    test_file = tdms.TdmsFile(DATA_DIR + '/raw1.tdms')
    obj_path = test_file.groups()[0]
    data = test_file.object(obj_path, 'First  Channel').data
    np.testing.assert_almost_equal(data[:10],
                                   [-0.18402661, 0.14801477, -0.24506363,
                                    -0.29725028, -0.20020142, 0.18158513,
                                    0.02380444, 0.20661031, 0.20447401,
                                    0.2517777])


def test_big_endian_format():
    """Test reading a file that encodes data in big endian mode"""
    test_file = tdms.TdmsFile(DATA_DIR + '/big_endian.tdms')
    data = test_file.object('Measured Data', 'Phase sweep').data
    np.testing.assert_almost_equal(data[:10],
                                   [0.0000000, 0.0634176, 0.1265799,
                                    0.1892325, 0.2511234, 0.3120033,
                                    0.3716271, 0.4297548, 0.4861524,
                                    0.5405928])
