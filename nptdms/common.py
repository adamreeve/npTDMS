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


class ObjectPath(object):
    """ Represents the path of an object in a TDMS file

        :ivar group: Group name or None for the root object
        :ivar channel: Channel name or None for the root object or a group objecct
    """
    def __init__(self, *path_components):
        self.group = None
        self.channel = None
        if len(path_components) > 0:
            self.group = path_components[0]
        if len(path_components) > 1:
            self.channel = path_components[1]
        if len(path_components) > 2:
            raise ValueError("Object path may only have up to two components")
        self._path = _components_to_path(self.group, self.channel)

    @property
    def is_root(self):
        return self.group is None

    @property
    def is_group(self):
        return self.group is not None and self.channel is None

    @property
    def is_channel(self):
        return self.channel is not None

    def group_path(self):
        """ For channel paths, returns the path of the channel's group as a string
        """
        return _components_to_path(self.group, None)

    @staticmethod
    def from_string(path_string):
        components = list(_path_components(path_string))
        return ObjectPath(*components)

    def __str__(self):
        """ String representation of the object path
        """
        return self._path


def _path_components(path):
    """ Generator that yields components within an object path
    """
    # Iterate over each character and the next character
    chars = zip_longest(path, path[1:])
    try:
        # Iterate over components
        while True:
            char, next_char = next(chars)
            if char != '/':
                raise ValueError("Invalid path, expected \"/\"")
            elif next_char is not None and next_char != "'":
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


def _components_to_path(group, channel):
    components = []
    if group is not None:
        components.append(group)
    if channel is not None:
        components.append(channel)
    return ('/' + '/'.join(
        ["'" + c.replace("'", "''") + "'" for c in components]))
