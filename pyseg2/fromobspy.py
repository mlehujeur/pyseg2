from pyseg2.seg2file import Seg2Trace, Seg2File
from pyseg2.binaryblocks import FileDescriptorSubBlock, TraceDescriptorSubBlock, FreeFormatSection, TraceDataBlock, Seg2String
from enum import Enum

class SEG2Sorting(Enum):
    AS_ACQUIRED = "AS_ACQUIRED"
    CDP_GATHER = "CDP_GATHER"
    CDP_STACK = "CDP_STACK"
    COMMON_OFFSET = "COMMON_OFFSET"
    COMMON_RECEIVER = "COMMON_RECEIVER"
    COMMON_SOURCE = "COMMON_SOURCE"

class SEG2Units(Enum):
    FEET = "FEET"
    METERS = "METERS"
    INCHES = "INCHES"
    CENTIMETERS = "CENTIMETERS"
    NONE = "NONE"


def pyseg2_from_obspy_stream(seg2file, obspy_stream, delay=0, stacks=1, line_id='', 
                             source_x=0, units: str = SEG2Units.METERS.value, 
                             sorting: str = SEG2Sorting.COMMON_SOURCE.value, company=''):
    """
    Populate seg2file structure with data and metadata from the Obspy stream.

    Parameters:
    -----------
    seg2file : [type]
        The SEG2 file structure to populate.
    obspy_stream : obspy.Stream
        The Obspy stream containing seismic trace data.
    delay : int, optional
        Delay applied to the traces, by default 0
    stacks : int, optional
        Number of stacks, by default 1
    line_id : str, optional
        Identifier for the seismic line, by default ''
    source_x : int, optional
        Source coordinate X, by default 0
    units : str, optional
        Units of measurement. Allowed values are:
        FEET, METERS, INCHES, CENTIMETERS, NONE (defined in "SEG2Units" enum). Default is METERS.
    sorting : str, optional
        Sorting method applied to the traces. Available options are:
        AS_ACQUIRED, CDP_GATHER, CDP_STACK, COMMON_OFFSET, COMMON_RECEIVER, COMMON_SOURCE  (defined in "SEG2Sorting" enum).
        Default is COMMON_SOURCE.
    company : str, optional
        Company name metadata, by default ''

    Raises:
    -------
    ValueError:
        If 'sorting' or 'units' values are not among the allowed enum options.
    """
    try:
        sorting = SEG2Sorting(sorting)
    except ValueError:
        raise ValueError(f"Invalid sorting method: {sorting}")

    try:
        units = SEG2Units(units)
    except ValueError:
        raise ValueError(f"Invalid units: {units}")

    date = obspy_stream[0].stats.starttime.strftime("%d/%m/%Y")
    time = obspy_stream[0].stats.starttime.strftime("%H:%M:%S")
    Units = 'METERS'
    DateString = Seg2String(parent = seg2file.free_format_section, text = f"ACQUISITION_DATE {date}")
    TimeString = Seg2String(parent = seg2file.free_format_section, text = f"ACQUISITION_TIME {time}")
    if company:
        CompanyString = Seg2String(parent = seg2file.free_format_section, text = f"COMPANY {company}")
        seg2file.free_format_section.strings.append(CompanyString)
    SortingString = Seg2String(parent = seg2file.free_format_section, text = f"TRACE_SORT {sorting}")
    UnitsString = Seg2String(parent = seg2file.free_format_section, text = f"UNITS {Units}")
    seg2file.free_format_section.strings.append(DateString)
    seg2file.free_format_section.strings.append(TimeString)
    seg2file.free_format_section.strings.append(SortingString)
    seg2file.free_format_section.strings.append(UnitsString)

    traces = []
    for i, trace in enumerate(obspy_stream):
        trace_descriptor_subblock = TraceDescriptorSubBlock(parent=seg2file.file_descriptor_subblock)
        trace_free_format_section = FreeFormatSection(parent=trace_descriptor_subblock)

        DelayString = Seg2String(parent = trace_free_format_section, text = f"DELAY {str(delay)}")
        trace_free_format_section.strings.append(DelayString) 
        
        if line_id:
            LineIDString = Seg2String(parent = trace_free_format_section, text = f"LINE_ID {str(line_id)}")
            trace_free_format_section.strings.append(LineIDString) 
        
        RxString = Seg2String(parent = trace_free_format_section, text = f"RECEIVER_LOCATION {str(trace.stats.distance)}")
        trace_free_format_section.strings.append(RxString) 
        
        RxNoString = Seg2String(parent = trace_free_format_section, text = f"RECEIVER_STATION_NUMBER {str(trace.stats.station)}")
        trace_free_format_section.strings.append(RxNoString) 
        
        SIntString = Seg2String(parent = trace_free_format_section, text = f"SAMPLE_INTERVAL {str(trace.stats.delta)}")
        trace_free_format_section.strings.append(SIntString)

        SxString = Seg2String(parent = trace_free_format_section, text = f"SOURCE_LOCATION {str(source_x)}")
        trace_free_format_section.strings.append(SxString)

        StackString = Seg2String(parent = trace_free_format_section, text = f"STACK {str((int)(stacks))}")
        trace_free_format_section.strings.append(StackString)

        trace_data_block = TraceDataBlock(parent=trace_descriptor_subblock)
        trace_data_block.data = trace.data 
        seg2trace = Seg2Trace(trace_descriptor_subblock, trace_free_format_section, trace_data_block)
        
        traces.append(seg2trace)

    seg2file.seg2traces = traces
