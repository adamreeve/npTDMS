""" Test reading example TDMS files """

import os
from io import BytesIO

import numpy as np
import pytest

from nptdms import tdms, TdmsWriter, TdmsFile

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/data'


def test_labview_file():
    """Test reading a file that was created by LabVIEW"""
    test_file = tdms.TdmsFile(DATA_DIR + '/Digital_Input.tdms')
    group = ("07/09/2012 06:58:23 PM - " +
             "Digital Input - Decimated Data_Level1")
    channel = "Dev1_port3_line7 - line 0"
    expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    data = test_file[group][channel].data
    np.testing.assert_almost_equal(data[:10], expected)


def test_raw_format():
    """Test reading a file with DAQmx raw data"""
    test_file = tdms.TdmsFile(DATA_DIR + '/raw1.tdms')
    group = test_file.groups()[0]
    data = group['First  Channel'].data
    np.testing.assert_almost_equal(data[:10],
                                   [-0.18402661, 0.14801477, -0.24506363,
                                    -0.29725028, -0.20020142, 0.18158513,
                                    0.02380444, 0.20661031, 0.20447401,
                                    0.2517777])


def tdms_files_assert_equal(tdms1: TdmsFile, tdms2: TdmsFile):
    """Assert that two TdmsFile instances are equal"""
    # verify file properties
    for p in tdms1.properties:
        np.testing.assert_equal(tdms1.properties[p], tdms2.properties[p])
    # verify group
    for group in tdms1.groups():
        # verify group properties
        for p in group.properties:
            np.testing.assert_equal(group.properties[p], tdms2[group.name].properties[p])
        # verify channels
        for channel in group.channels():
            # verify channel data
            np.testing.assert_equal(channel.data, tdms2[group.name][channel.name].data)
            # verify channel properties
            for p in channel.properties:
                np.testing.assert_equal(channel.properties[p], tdms2[group.name][channel.name].properties[p])


@pytest.mark.parametrize("tdms_file", [
    'raw_timestamps.tdms',  # Test defragmentation of a file with raw timestamps
    # 'raw1.tdms', # <- cannot defragment this file (ValueError: Channel data must be a 1d array)
    'Digital_Input.tdms',
    'big_endian.tdms',
])
def test_defragment(tdms_file):
    """Test defragmentation round trip for a TDMS file"""
    test_file_path = DATA_DIR + '/' + tdms_file
    output_file = BytesIO()

    # verify we can defragment a file with raw timestamps
    TdmsWriter.defragment(test_file_path, output_file)

    # rewind output file BytesIO instance, so it can read it back in as a TdmsFile
    output_file.seek(0)

    # verify that both TdmsFile objects are the same
    tdms_files_assert_equal(
        tdms.TdmsFile(test_file_path, raw_timestamps=True),
        tdms.TdmsFile(output_file, raw_timestamps=True),
    )


def test_big_endian_format():
    """Test reading a file that encodes data in big endian mode"""
    test_file = tdms.TdmsFile(DATA_DIR + '/big_endian.tdms')
    data = test_file['Measured Data']['Phase sweep'].data
    np.testing.assert_almost_equal(data[:10],
                                   [0.0000000, 0.0634176, 0.1265799,
                                    0.1892325, 0.2511234, 0.3120033,
                                    0.3716271, 0.4297548, 0.4861524,
                                    0.5405928])
