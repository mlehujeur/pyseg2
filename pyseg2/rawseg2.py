from typing import Tuple, List
import os
import numpy as np

from pyseg2.seg2file import Seg2File, Seg2Trace
from pyseg2.binaryblocks import \
    Seg2String, FreeFormatSection, \
    TraceDescriptorSubBlock, TraceDataBlock


def dict_recurs(dictionnary, _parents=None):
    """explore a dictionary recursively"""
    if _parents is None:
        _parents = []
    
    for key, val in dictionnary.items():   
        if isinstance(val, dict):
            for key, val in dict_recurs(val, _parents=_parents + [key]):
                yield key, val
        else:
            yield ".".join(_parents + [key]), val
            


def string2value(value: str):
    """
    try to recover data types
    assuming that type_names are included in the strings
    """
    if value.startswith(('int(', 'str(', 'float(', 'complex(')):
        value = eval(value)

    elif value.startswith(('list(',)):
        value = eval(value)

    elif value.startswith(('ndarray(',)):
        ndarray = np.array
        value = eval(value.replace(' ', ','))

    else:
        pass

    return value


def value2string(value, include_type_names):
    if include_type_names:
        return f"{type(value).__name__}({value})"
    else:
        return value


def write_raw_seg2(
    filename: str, 
    file_header: dict,
    trace_header_and_data: List[Tuple[dict, np.ndarray]],
    allow_overwrite: bool=False,
    include_type_names: bool=False):
    """
    write a raw seg2 file
    Note: obspy might not be able to read it if some required fields
    are missing like DELAY, SAMPLING_INTERVAL, ...

    :param filename: name of the seg2 file to write
    :param file_header: a dictionary to store in the file header
    :param trace_header_and_data: a list of tuples, with the trace header and data array
    :param allow_overwrite: to allow overwriting of existing file
    :return seg2: the Seg2File instance writen
    """
    if os.path.isfile(filename):
        if not allow_overwrite:
            raise IOError(f"{filename} exists")

    seg2 = build_raw_seg2(file_header, include_type_names, trace_header_and_data)

    with open(filename, 'wb') as fid:
        fid.write(seg2.pack())
            
    return seg2


def build_raw_seg2(file_header, include_type_names, trace_header_and_data):
    """

    :param file_header:
    :param include_type_names:
    :param trace_header_and_data:
    :return seg2: the Seg2File instance to write
    """
    seg2 = Seg2File()
    seg2.free_format_section.strings = []
    for key, value in dict_recurs(file_header):

        string = Seg2String(
            parent=seg2.file_descriptor_subblock,
            text=f"{key} {value2string(value, include_type_names)}")

        seg2.free_format_section.strings.append(string)

    for trace_header, trace_data in trace_header_and_data:
        assert isinstance(trace_header, dict)
        assert isinstance(trace_data, np.ndarray)

        trace_descriptor_subblock = \
            TraceDescriptorSubBlock(
                parent=seg2.file_descriptor_subblock)

        trace_free_format_section = \
            FreeFormatSection(
                parent=trace_descriptor_subblock,
                strings=[])

        for key, value in dict_recurs(trace_header):

            string = Seg2String(
                parent=trace_descriptor_subblock,
                text=f"{key} {value2string(value, include_type_names)}")

            trace_free_format_section.strings.append(string)

        trace_data_block = TraceDataBlock(
            parent=trace_descriptor_subblock,
        )

        seg2trace = Seg2Trace(
            trace_descriptor_subblock=trace_descriptor_subblock,
            trace_free_format_section=trace_free_format_section,
            trace_data_block=trace_data_block,
        )

        seg2trace.trace_data_block.data = trace_data

        seg2.seg2traces.append(seg2trace)
    return seg2


def read_raw_seg2(filename: str, evaluate_types: bool=False):
    """
    :param filename:
    :param evaluate_types:
        try to evaluate the content of the fields
        to recover their types
        to be used with include_type_names=True in build_raw_seg2
    :return :
    """

    seg2 = Seg2File()
    with open(filename, 'rb') as fid:
        seg2.load(fid)

    file_header = {}

    for string in seg2.free_format_section.strings:
        value = string.value

        if evaluate_types:
            value = string2value(value)

        file_header[string.key] = value

    trace_header_and_data = []
    for seg2trace in seg2.seg2traces:
        trace_data = seg2trace.trace_data_block.data

        trace_header = {}
        for string in seg2trace.trace_free_format_section.strings:
            value = string.value

            if evaluate_types:
                value = string2value(value)

            trace_header[string.key] = value

        trace_header_and_data.append((trace_header, trace_data))

    return file_header, trace_header_and_data


if __name__ == "__main__":

    file_header = {
        "I": 0,
        "II": "0",
        "III": 0.,
        "IV": [0, "0", 0., np.float64(1.), np.float32(0.), 0.j],
        "V": np.arange(3),
        "VI": {
            "1": 1,
            "2": {"a": 2}
            }
        }

    header1 = {"AAAAAAAAAAAAAAAAAA": 10, "BBBBBBBBBBBBB": 20, "SAMPLE_INTERVAL": 0.1}
    data1 = np.arange(3).astype('float64')

    header2 = {"AAAAAAAAAAAAAAAAAA": 100, "BBBBBBBBBBBBB": 200, "SAMPLE_INTERVAL": 0.2}
    data2 = np.arange(4).astype('float32')

    write_raw_seg2(
        filename="toto.seg2",
        file_header=file_header,
        trace_header_and_data=[
            (header1, data1),
            (header2, data2)],
        allow_overwrite=True,
        include_type_names=True)

    file_header, trace_header_and_data = read_raw_seg2("toto.seg2", evaluate_types=True)

    print(file_header)
    for header, data in trace_header_and_data:
        print(header)
        print(data, data.dtype)
