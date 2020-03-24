import os
from nptdms.tdmsinfo import tdmsinfo
from nptdms.test.util import GeneratedFile, basic_segment


def test_tdmsinfo():
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        tdmsinfo(temp_file.name)
    finally:
        os.remove(temp_file.name)


def test_tdmsinfo_with_properties():
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        tdmsinfo(temp_file.name, show_properties=True)
    finally:
        os.remove(temp_file.name)
