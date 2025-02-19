"""
PySeg2
@copyright : Maximilien Lehujeur
2023/05/12
"""

from typing import List, Tuple, Dict
import os, datetime, warnings
import numpy as np

from pyseg2.binaryblocks import (
    TraceDataBlock, Seg2String, TraceDescriptorSubBlock, FreeFormatSection, TracePointerSubblock)

from pyseg2.seg2file import Seg2Trace, Seg2File

try:
    from obspy.core import Stream, Trace, AttribDict, UTCDateTime
    OBSPY_AVAILABLE = True

except ImportError:
    OBSPY_AVAILABLE = False


def _extract_text_from_obspy_seg2_dict(
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
        if isinstance(val, list):
            val = line_terminator.join([str(_) for _ in val])
        text = f"{key.upper()} {val}"
        texts.append(text)
    return texts


def statsrecurs(stats: AttribDict, path=""):
    """recursive iterator on a stats object
    (used by obspy to store stream or trace header metadata) """

    for key, val in stats.items():
        if isinstance(val, AttribDict):
            for subkey, subval in statsrecurs(stats=val, path=path + key + "."):
                yield subkey, subval
        else:
            yield path + key, val


def _extract_text_from_obspy_noseg2_dict(
        stats: "obspy.core.AttribDict", line_terminator: str) -> List[str]:
    """
    :param stats:
        obspy.core.stream.Stream.stats or
        obspy.core.trace.Trace.stats
    :param line_terminator: the terminating character to use
    :return texts: the list of str to put in Seg2String objects
    """

    texts = []
    for key, val in statsrecurs(stats):
        if key.lower().startswith('seg2.'):
            # not handled in this function
            continue

        if key.lower() in ["delta", "sampling_rate", "starttime", "endtime", "calib",
                           "network", "station", "location", "channel"]:
            # ignore the default obspy fields
            continue

        if isinstance(val, list):
            val = line_terminator.join([str(_) for _ in val])
        text = f"noseg2.{key} {val}"
        texts.append(text)
    return texts


def obspy_to_pyseg2(stream: "obspy.core.stream.Stream") -> Seg2File:
    """

    :param stream:
    :return:
    """
    seg2 = Seg2File()

    line_terminator = seg2.file_descriptor_subblock.line_terminator.decode('ascii')
    seg2.free_format_section.strings = []

    # ======== Export the stream.stats.seg2 dictionary
    try:
        stream.stats['seg2']  # try to access this field
        stream_stats = stream.stats

    except AttributeError:
        # no stats at all
        stream_stats = {'seg2': {}}

    except KeyError:
        # stats exists but no seg2 dictionary
        stream_stats = stream.stats
        stream_stats["seg2"] = {}

    for text in _extract_text_from_obspy_seg2_dict(
            stats=stream_stats['seg2'],
            line_terminator=line_terminator):

        string = Seg2String(
            parent=seg2.file_descriptor_subblock,
            text=text,
            )
        seg2.free_format_section.strings.append(string)

    # ======== Export the other fields from the stream.stats header (noseg2)
    for text in _extract_text_from_obspy_noseg2_dict(
        stats=stream.stats,
        line_terminator=line_terminator):

        string = Seg2String(
            parent=seg2.file_descriptor_subblock,
            text=text,
            )
        seg2.free_format_section.strings.append(string)

    for trace in stream:
        trace: "obspy.core.trace.Trace"

        # ======== Export the trace.stats.seg2 dictionary
        # update the seg2 header from the conventional fields
        try:
            # do NOT use hasattr !!!
            trace.stats['seg2']
        except KeyError:
            trace.stats['seg2'] = {}

        # force ce file sample_interval (seg2 convention) to the value of the dynamic obspy field delta
        trace.stats['seg2']['SAMPLE_INTERVAL'] = trace.stats.delta
        #TODO : starttime? ...

        trace_descriptor_subblock = \
            TraceDescriptorSubBlock(
                parent=seg2.file_descriptor_subblock)

        trace_free_format_section = \
            FreeFormatSection(
                parent=trace_descriptor_subblock,
                strings=[])

        for text in _extract_text_from_obspy_seg2_dict(
                stats=trace.stats['seg2'], line_terminator=line_terminator):
            string = Seg2String(
                parent=trace_descriptor_subblock,
                text=text)
            trace_free_format_section.strings.append(string)

        for text in _extract_text_from_obspy_noseg2_dict(
                stats=trace.stats, line_terminator=line_terminator):
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

    return seg2


def pyseg2_to_obspy(seg2: Seg2File, **kwargs) -> [Dict, List[Tuple[Dict, np.ndarray]]]:

    """
    :param seg2: a Seg2File object loaded with binary data buffer
    :return stream_stats: a dictionary to be stored as AttribDict at the stream level
    :return traces_stats_and_data: a list of tuples (trace_stats, trace_data)
        where trace_stats is a dictionary to be used to generate the "stats" attributes of the obspy traces
        where trace_data is a numpy array to be used for the data array of each trace

    """
    for key in kwargs.keys():
        warnings.warn(f'WARNING keyword argument {key=} is ignored for now')

    stream_stats = {"seg2": {}, }

    for string in seg2.free_format_section.strings:
        stream_stats['seg2'][string.key] = string.value
        # if string.key == "NOTE":
        #     for item in string.value.split(';'):
        #         key = item.split()[0]
        #         value = item.split(key)[-1]
        #         stream_stats['seg2'][key] = value.strip()
        # else:
        #     stream_stats['seg2'][string.key] = string.value

    # default starttime
    year, month, day, hour, minute, second, microsecond = 1970, 1, 1, 0, 0, 0, 0

    # search for trace time in the file header
    for string in seg2.free_format_section.strings:
        if string.key.upper() == "ACQUISITION_DATE":
            # assume format yyyy/mm/dd
            year, month, day = [int(_) for _ in string.value.split('/')]

        elif string.key.upper() == "ACQUISITION_TIME":
            # assume format hh:mm:ss
            hour, minute, second = [int(_) for _ in string.value.split(':')]

    # interpret the values found in term of starttime
    try:
        starttime = datetime.datetime(year, month, day, hour, minute, second)
    except ValueError as err:
        if "day is out of range for month" in str(err):
            # may be the time format is not yyyy/mm/dd, try dd/mm/yyyy
            year, day = day, year
            starttime = datetime.datetime(year, month, day, hour, minute, second)
                                           
    traces_stats_and_data = []
    for seg2trace in seg2.seg2traces:
        trace_data = seg2trace.trace_data_block.data
        trace_stats = {
            "network": "",
            "station": "",
            "location": "",
            "channel": "",
            "npts": len(trace_data),
            "delta": 1.0,
            "starttime": starttime,
            "calib": 1.,
            "seg2": {}}

        _year = trace_stats['starttime'].year
        _month = trace_stats['starttime'].month
        _day = trace_stats['starttime'].day
        _hour = trace_stats['starttime'].hour
        _minute = trace_stats['starttime'].minute
        _second = trace_stats['starttime'].second

        for string in seg2trace.trace_free_format_section.strings:
            trace_stats['seg2'][string.key] = string.value
            if string.key.upper() == "SAMPLE_INTERVAL":
                trace_stats['delta'] = float(string.value)

            elif string.key.upper() == "ACQUISITION_DATE":
                _year, _month, _day = [int(_) for _ in string.value.split('/')]

            elif string.key.upper() == "ACQUISITION_TIME":
                # assume format hh:mm:ss
                _hour, _minute, _second = [int(_) for _ in string.value.split(':')]

        trace_stats['starttime'] = datetime.datetime(_year, _month, _day, _hour, _minute, _second)
        traces_stats_and_data.append((trace_stats, trace_data))

    return stream_stats, traces_stats_and_data


def pyseg2_to_obspy_stream(seg2: Seg2File, **kwargs) -> "obspy.core.stream.Stream":
    if not OBSPY_AVAILABLE:
        raise Exception('Obpsy needed in this function')

    stream_stats, traces_stats_and_data = pyseg2_to_obspy(seg2=seg2, **kwargs)

    stream = Stream()
    stream.stats = AttribDict(stream_stats)

    for trace_stats, trace_data in traces_stats_and_data:
        trace = Trace(header=AttribDict(trace_stats), data=trace_data)
        trace.stats.starttime = UTCDateTime(trace.stats.starttime)
        stream.append(trace)

    return stream


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

    print(stream.stats, stream[0].stats)

    seg2 = obspy_to_pyseg2(stream)

    with open(filename, 'wb') as fil:
        fil.write(seg2.pack())


if __name__ == '__main__':
    # from obspy import read
    #
    # st = read('./toto.seg2')
    #
    # os.system('rm -f titi.seg2')
    # write_obspy_stream_as_seg2(stream=st, filename="titi.seg2")

    from pyseg2.toobspy import pyseg2_to_obspy
    from pyseg2.seg2file import Seg2File
    seg2 = Seg2File()

    f = "../../readdat/readdat/filesamples/seg2file_cdz.sg2"
    with open(f, 'rb') as fid:
        seg2.load(fid)

    stream_stats, traces_stats_and_data = pyseg2_to_obspy(seg2)
    from obspy.core import Stream, Trace
    stream = Stream()
    stream.stats = stream_stats
    for trace_stats, trace_data in traces_stats_and_data:
        trace = Trace(header=trace_stats, data=trace_data)
        stream.append(trace)
    print(stream)




