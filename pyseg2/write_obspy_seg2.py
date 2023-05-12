"""
PySeg2
@copyright : Maximilien Lehujeur
2023/05/12
"""

from typing import List
import os
from pyseg2.binaryblocks import TraceDataBlock, Seg2String, TraceDescriptorSubBlock, FreeFormatSection, TracePointerSubblock
from pyseg2.seg2file import Seg2Trace, Seg2File


def _extract_text_from_obspy_dict(
        stats: "obspy.core.AttribDict", line_terminator: str) -> List[str]:
    """
    :param stats:
        obspy.core.stream.Stream.stats.seg2 or
        obspy.core.trace.Trace.stats.seg2
    :param line_terminator: the terminating character to use
    :return texts: the list of str to put in Seg2String objects
    """

    texts = []
    for key, val in stats.items():
        if key == "NOTE" or isinstance(val, list):
            val = line_terminator.join(val)
        text = f"{key.upper()} {val}"
        texts.append(text)
    return texts


def write_obspy_stream_as_seg2(
        stream: "obspy.core.stream.Stream",
        filename: str):
    """
    :param stream: obspy.core.stream.Stream object read from seg2 file
    :param filename: name of file to write
    :return:
    """

    if os.path.exists(filename):
        raise IOError(filename)

    assert filename.upper().endswith('SG2') or \
           filename.upper().endswith('SEG2')

    seg2 = Seg2File()
    line_terminator = seg2.file_descriptor_subblock.line_terminator.decode('ascii')

    seg2.free_format_section.strings = []
    for text in _extract_text_from_obspy_dict(stats=stream.stats.seg2, line_terminator=line_terminator):
        string = Seg2String(
            parent=seg2.file_descriptor_subblock,
            text=text,
            )
        seg2.free_format_section.strings.append(string)

    for trace in stream:
        trace: "obspy.core.trace.Trace"

        trace_descriptor_subblock = \
            TraceDescriptorSubBlock(
                parent=seg2.file_descriptor_subblock)

        trace_free_format_section = \
            FreeFormatSection(
                parent=trace_descriptor_subblock,
                strings=[])

        for text in _extract_text_from_obspy_dict(stats=trace.stats.seg2, line_terminator=line_terminator):
            string = Seg2String(
                parent=trace_descriptor_subblock,
                text=text)
            trace_free_format_section.strings.append(string)

        trace_data_block = TraceDataBlock(
            parent=trace_descriptor_subblock)

        seg2trace = Seg2Trace(
            trace_descriptor_subblock=trace_descriptor_subblock,
            trace_free_format_section=trace_free_format_section,
            trace_data_block=trace_data_block,
            )

        trace_data_block.data = trace.data
        seg2.seg2traces.append(seg2trace)

    with open(filename, 'wb') as fil:
        fil.write(seg2.pack())


if __name__ == '__main__':
    from obspy import read

    st = read('./toto.seg2')

    os.system('rm -f titi.seg2')
    write_obspy_stream_as_seg2(stream=st, filename="titi.seg2")