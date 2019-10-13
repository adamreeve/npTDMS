import itertools
try:
    long
except NameError:
    # Python 3
    long = int
try:
    zip_longest = itertools.izip_longest
except AttributeError:
    # Python 3
    zip_longest = itertools.zip_longest


toc_properties = {
    'kTocMetaData': (long(1) << 1),
    'kTocRawData': (long(1) << 3),
    'kTocDAQmxRawData': (long(1) << 7),
    'kTocInterleavedData': (long(1) << 5),
    'kTocBigEndian': (long(1) << 6),
    'kTocNewObjList': (long(1) << 2)
}


def path_components(path):
    """Convert a path into group and channel name components"""

    def yield_components(path):
        # Iterate over each character and the next character
        chars = zip_longest(path, path[1:])
        try:
            # Iterate over components
            while True:
                char, next_char = next(chars)
                if char != '/':
                    raise ValueError("Invalid path, expected \"/\"")
                elif (next_char is not None and next_char != "'"):
                    raise ValueError("Invalid path, expected \"'\"")
                else:
                    # Consume "'" or raise StopIteration if at the end
                    next(chars)
                component = []
                # Iterate over characters in component name
                while True:
                    char, next_char = next(chars)
                    if char == "'" and next_char == "'":
                        component += "'"
                        # Consume second "'"
                        next(chars)
                    elif char == "'":
                        yield "".join(component)
                        break
                    else:
                        component += char
        except StopIteration:
            return

    return list(yield_components(path))
