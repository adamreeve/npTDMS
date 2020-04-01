from __future__ import print_function

from argparse import ArgumentParser
import logging

from nptdms import TdmsFile
from nptdms.log import log_manager


def main():
    parser = ArgumentParser(
        description="List the contents of a LabView TDMS file.")
    parser.add_argument(
        '-p', '--properties', action="store_true",
        help="Include channel properties.")
    parser.add_argument(
        '-d', '--debug', action="store_true",
        help="Print debugging information to stderr.")
    parser.add_argument(
        'tdms_file',
        help="TDMS file to read.")
    args = parser.parse_args()

    if args.debug:
        log_manager.set_level(logging.DEBUG)

    tdmsinfo(args.tdms_file, args.properties)


def tdmsinfo(file, show_properties=False):
    tdms_file = TdmsFile.read_metadata(file)

    level = 0
    display('/', level)
    if show_properties:
        display_properties(tdms_file, level + 1)
    for group in tdms_file.groups():
        level = 1
        display("%s" % group.path, level)
        if show_properties:
            display_properties(group, level + 1)
        for channel in group.channels():
            level = 2
            display("%s" % channel.path, level)
            if show_properties:
                level = 3
                if channel.data_type is not None:
                    display("data type: %s" % channel.data_type.__name__, level)
                display("length: %d" % len(channel), level)
                display_properties(channel, level)


def display_properties(tdms_object, level):
    if tdms_object.properties:
        display("properties:", level)
        for prop, val in tdms_object.properties.items():
            display("%s: %s" % (prop, val), level + 1)


def display(s, level):
    print("%s%s" % (" " * 2 * level, s))
