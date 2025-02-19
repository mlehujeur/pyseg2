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
            

def write_raw_seg2(
    filename: str, 
    file_header: dict,
    trace_header_and_data: List[Tuple[dict, np.ndarray]],
    allow_overwrite: bool=False,
    include_type_names: bool=False):
    """

    :param filename:
    :param file_header:
    :param trace_header_and_data:
    :param allow_overwrite:
    :return:
    """
    if os.path.isfile(filename):
        if not allow_overwrite:
            raise IOError(f"{filename} exists")
            
    seg2 = Seg2File()

    seg2.free_format_section.strings = []
    for key, value in dict_recurs(file_header):
        if include_type_names:
            # string = Seg2String(
            #      parent=seg2.file_descriptor_subblock,
            #      text=f"{key} {value.__class__.__name__}({repr(value)})")
            raise NotImplementedError
        else:
            string = Seg2String(
                parent=seg2.file_descriptor_subblock,
                text=f"{key} {value}")

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
                text=f"{key} {repr(value)}")    
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

    with open(filename, 'wb') as fid:
        fid.write(seg2.pack())
            
    return

def read_raw_seg2(filename: str):
    seg2 = Seg2File()
    with open(filename, 'rb') as fid:
        seg2.load(fid)

    file_header = {}

    for string in seg2.free_format_section.strings:
        file_header[string.key] = string.value

    trace_header_and_data = []
    for seg2trace in seg2.seg2traces:
        trace_data = seg2trace.trace_data_block.data

        trace_header = {}
        for string in seg2trace.trace_free_format_section.strings:
            trace_header[string.key] = string.value

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
        include_type_names=False)

    file_header, trace_header_and_data = read_raw_seg2("toto.seg2")

    print(file_header)
    for header, data in trace_header_and_data:
        print(header)
        print(data, data.dtype)
