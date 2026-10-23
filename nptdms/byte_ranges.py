""" Read objects from part of a TDMS file using the byte ranges they are stored in
"""


def merge_byte_ranges(*byte_ranges):
    """ Merge lists of byte ranges, for example to combine the byte ranges of several objects

    :param byte_ranges: Lists of ``(offset, length)`` byte ranges.
    :returns: Sorted list of ``(offset, length)`` tuples, with overlapping and adjacent ranges merged.
    """
    merged = []
    for (offset, length) in sorted(tuple(r) for ranges in byte_ranges for r in ranges):
        if merged and offset <= merged[-1][0] + merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset + length - merged[-1][0]))
        else:
            merged.append((offset, length))
    return merged


def extract_segments(file, byte_ranges):
    """ Read byte ranges of whole segments from a TDMS file and concatenate them into new TDMS data

    When these are the byte ranges of an object, the object can be read from the result,
    for example with ``TdmsFile.read(io.BytesIO(data))``.

    :param file: Path to a TDMS data file, or an open binary file.
    :param byte_ranges: List of ``(offset, length)`` byte ranges.
    :rtype: bytes
    """
    if not hasattr(file, 'read'):
        with open(file, 'rb') as f:
            return extract_segments(f, byte_ranges)
    data = []
    for (offset, length) in merge_byte_ranges(byte_ranges):
        file.seek(offset)
        data.append(file.read(length))
        if len(data[-1]) != length or data[-1][:4] != b'TDSm':
            raise ValueError("Data at position %d is not a TDMS segment of %d bytes" % (offset, length))
    return b''.join(data)
