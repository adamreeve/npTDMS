import os
import sys
from nptdms import tdmsinfo
from nptdms.test.util import GeneratedFile, basic_segment
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


def test_tdmsinfo(capsys):
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with patch.object(sys, 'argv', ['tdmsinfo.py', temp_file.name]):
            tdmsinfo.main()
            captured = capsys.readouterr()
            assert "/'Group'/'Channel1'" in captured.out
            assert "wf_start_offset" not in captured.out
    finally:
        os.remove(temp_file.name)


def test_tdmsinfo_with_properties(capsys):
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with patch.object(sys, 'argv', ['tdmsinfo.py', temp_file.name, '--properties']):
            tdmsinfo.main()
            captured = capsys.readouterr()
            assert "/'Group'/'Channel1'" in captured.out
            assert "wf_start_offset: 0.0" in captured.out
            assert "length: 2" in captured.out
    finally:
        os.remove(temp_file.name)


def test_tdmsinfo_with_debug_output(caplog):
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with patch.object(sys, 'argv', ['tdmsinfo.py', temp_file.name, '--debug']):
            tdmsinfo.main()
            assert "Reading metadata for object /'Group'/'Channel1'" in caplog.text
    finally:
        os.remove(temp_file.name)
