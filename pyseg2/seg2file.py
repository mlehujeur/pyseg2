"""
PySeg2
@copyright : Maximilien Lehujeur


The dataclasses are organized hierarchically
    so that modifying some attributes of the root classes (like endian)
    also modifies all children classes

The hierarchy of the classes is as follow (nb, this is not inheritance)
    FileDescriptorSubBlock
        TracePointerSubblock
            FreeFormatSection
            Seg2String
        TraceDescriptorSubBlock
            FreeFormatSection
            TraceDataBlock
        
Warning:
    loading data: the controlling parameters (e.g. num of traces)
                  are read in the parent binary blocks and used by childrens to load data
                  => size methods are supposed to give the expected size of the object in bytes
                     according to headers
    packing data: the controlling parameters are inferred from the current state of the variables
                  and must be updated in the parent classes
                  => number_of_bytes methods should provide the actual number of bytes of an object 
    
"""

from typing import List
from dataclasses import dataclass
import numpy as np

from pyseg2.binaryblocks import \
    FileDescriptorSubBlock, TracePointerSubblock, \
    FreeFormatSection, \
    TraceDescriptorSubBlock, TraceDataBlock


@dataclass
class Seg2Trace:
    trace_descriptor_subblock: TraceDescriptorSubBlock
    trace_free_format_section: FreeFormatSection
    trace_data_block: TraceDataBlock

    def pack(self):
        # pack the free format section of the trace descriptor
        # must be done first because the size must be adjusted in the trace_descriptor_subblock
        free_format_section_buffer = self.trace_free_format_section.pack()
        assert len(free_format_section_buffer) == self.trace_free_format_section.number_of_bytes()

        # update the field in the trace_descriptor_subblock
        self.trace_descriptor_subblock.size_of_descriptor_block = \
            self.trace_descriptor_subblock.number_of_bytes() + \
            self.trace_free_format_section.number_of_bytes()

        # now pack the trace_descriptor_subblock
        trace_description_buffer = self.trace_descriptor_subblock.pack()
        assert self.trace_descriptor_subblock.size_of_descriptor_block == \
               len(trace_description_buffer) + len(free_format_section_buffer)

        # finally, pack the data trace
        trace_data_buffer = self.trace_data_block.pack()

        return trace_description_buffer + free_format_section_buffer + trace_data_buffer

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        return self.trace_descriptor_subblock.number_of_bytes() + \
            self.trace_free_format_section.number_of_bytes() + \
            self.trace_data_block.number_of_bytes()


class Seg2File:
    def __init__(self, filename: str):
        """
        :param filename: name of seg2 file to read
        """
        self.file_descriptor_subblock = FileDescriptorSubBlock()

        with open(filename, 'rb') as fid:
            self.file_descriptor_subblock.load(fid)

            # self.trace_pointer_subblock.set(self.file_descriptor_subblock)
            self.trace_pointer_subblock = TracePointerSubblock(parent=self.file_descriptor_subblock)
            self.trace_pointer_subblock.load(fid)

            self.free_format_section = FreeFormatSection(parent=self.trace_pointer_subblock)
            self.free_format_section.load(fid)

            self.seg2traces: List[Seg2Trace] = []
            for n, trace_pointer in enumerate(self.trace_pointer_subblock.trace_pointers):
                # make sure the cursor is positioned at the beginning
                # of the trace
                fid.seek(trace_pointer, 0)

                trace_descriptor_subblock = TraceDescriptorSubBlock(parent=self.file_descriptor_subblock)
                trace_descriptor_subblock.load(fid)

                trace_free_format_section = FreeFormatSection(parent=trace_descriptor_subblock)
                trace_free_format_section.load(fid)

                trace_data_block = TraceDataBlock(parent=trace_descriptor_subblock)
                print(trace_data_block)
                print(fid.tell())

                trace_data_block.load(fid)

                seg2trace = Seg2Trace(
                    trace_descriptor_subblock=trace_descriptor_subblock,
                    trace_free_format_section=trace_free_format_section,
                    trace_data_block=trace_data_block)

                self.seg2traces.append(seg2trace)

    def __str__(self):
        s = f"# ============ File Descriptor\n"
        s += f"# ===== File Descriptor\n"
        s += f"{self.file_descriptor_subblock}\n"
        s += f"# =====\n"
        s += f"{self.trace_pointer_subblock}\n"
        s += f"# ============ Free Format Section\n"
        s += f"{self.free_format_section}\n"
        for n, trace in enumerate(self.seg2traces):
             s += f"# ============ Trace # {n}\n"
             s += str(trace) + "\n"
        return s

    def pack(self) -> bytes:
        """
        WARNING : the controlling parameters
                  used to pack data in binary format
                  are taken from the current state of the attributes
                  they must be updated in the parent objects so that
                  the header blocks match with the data blocks!
        :return buff: a buffer of bytes to save to file
        """
        # ==== recompute the trace pointer table
        number_of_traces = len(self.seg2traces)

        # update the file descriptor subblock
        # doing that, the children classes will be modified as well
        # It is crutial to fill it first otherwise the computed buffer sizes might be wrong
        self.file_descriptor_subblock.number_of_traces = number_of_traces  # N
        self.file_descriptor_subblock.size_of_trace_pointer_subblock = \
            4 * number_of_traces  # use the minimal value for M (might be higher)

        self.trace_pointer_subblock.trace_pointers = \
            np.empty(number_of_traces, dtype=np.uint32)

        # the file descriptor block depends on the trace pointer table
        size_of_file_descriptor_block = \
            self.file_descriptor_subblock.number_of_bytes() + \
            self.trace_pointer_subblock.number_of_bytes() + \
            self.free_format_section.number_of_bytes()

        self.trace_pointer_subblock.trace_pointers[0] = size_of_file_descriptor_block  # number of bytes in the header
        for n, trace in enumerate(self.seg2traces[:-1]):
            # put the number of bytes of the trace for now
            # x = self.trace_pointer_subblock.trace_pointers.dtype.type(trace.nbytes())
            self.trace_pointer_subblock.trace_pointers[n+1] = trace.number_of_bytes()

        # convert number of bytes into positions
        self.trace_pointer_subblock.trace_pointers = \
            self.trace_pointer_subblock.trace_pointers\
                .cumsum()\
                .astype('uint32')

        buff = self.file_descriptor_subblock.pack()
        buff += self.trace_pointer_subblock.pack()
        buff += self.free_format_section.pack()
        assert len(buff) == size_of_file_descriptor_block

        for n, trace in enumerate(self.seg2traces):
            # the current position must agree with the trace_pointer_table
            assert len(buff) == self.trace_pointer_subblock.trace_pointers[n]

            buff += trace.pack()
            print("packing", n, len(buff))

        return buff


if __name__ == "__main__":
    print('load toto')
    seg2 = Seg2File('./toto.seg2')

    seg2.seg2traces = seg2.seg2traces[:2]

    print('write tata')
    with open('tata.seg2', 'wb') as fil:
        fil.write(seg2.pack())

    print('load tata')
    seg2re = Seg2File('./tata.seg2')
    print(seg2re.seg2traces[0].trace_free_format_section)

    # exit()
    #
    # print('load tata')
    # with open('tata.seg2', 'rb') as fid:
    #     file_descriptor_subblock = FileDescriptorSubBlock()
    #     file_descriptor_subblock.load(fid)
    #
    #     trace_pointer_subblock = TracePointerSubblock(parent=file_descriptor_subblock)
    #     trace_pointer_subblock.load(fid)
    #
    #     free_format_section = FreeFormatSection(parent=trace_pointer_subblock)
    #     free_format_section.load(fid)
    #
    #     fid.seek(trace_pointer_subblock.trace_pointers[0], 0)
    #     trace_descriptor_subblock = TraceDescriptorSubBlock(parent=file_descriptor_subblock)
    #     trace_descriptor_subblock.load(fid)
    #
    #     trace_free_format_section = FreeFormatSection(parent=trace_descriptor_subblock)
    #     trace_free_format_section.load(fid)
    #
    #     trace_data_block = TraceDataBlock(parent=trace_descriptor_subblock)
    #     print("gggggggggg", fid.tell(), trace_pointer_subblock.trace_pointers[0])
    #     trace_data_block.load(fid)
    #
    #     # print(fid.tell())
    #     # trace_data_block.load(fid)
    #     # print(fid.tell())
    #
    #
    #
    # # print(seg2.file_descriptor_subblock)
    # # print(file_descriptor_subblock)
    #
    # # print(seg2.trace_pointer_subblock)
    # # print(trace_pointer_subblock)
    # # print(np.abs(seg2.trace_pointer_subblock.trace_pointers - trace_pointer_subblock.trace_pointers).sum())
    #
    # # print(seg2.free_format_section.nbytes())
    # # print(free_format_section.nbytes())
    # # for string in seg2.free_format_section.strings:
    # #     print(string)
    # # print()
    # # for string in free_format_section.strings:
    # #     print(string)
    #
    # # print(str(seg2.seg2traces[0].trace_descriptor_subblock).replace(',', ',\n\t'))
    # # print(str(trace_descriptor_subblock).replace(',', ',\n\t'))
    # #
    # # for string in seg2.seg2traces[0].trace_free_format_section.strings:
    # #     print(string)
    # # print()
    # # for string in trace_free_format_section.strings:
    # #     print(string)
    #
    # print(seg2.seg2traces[0].trace_data_block.data.shape)
    # print(trace_data_block.data.shape)
    # plt.figure()
    # plt.plot(seg2.seg2traces[0].trace_data_block.data)
    # plt.plot(trace_data_block.data)
    # plt.show()
    #
    # # seg2re = Seg2File('./tata.seg2')
    # # print(seg2re.seg2traces[0].trace_descriptor_block.free_format_section)
    #
    #
    #
    #
    #
    #
    # # print(seg2.file_descriptor_subblock)
    # # print(seg2.file_descriptor_subblock.pack())
    # #
    # # print(seg2.trace_pointer_subblock)
    # # print(seg2.trace_pointer_subblock.pack())
    #
    # # print(seg2.free_format_section)
    # # print(seg2.free_format_section.pack())
    #
    # # print(seg2.seg2traces[0].trace_descriptor_block.pack())
    # # print(seg2.seg2traces[0].trace_data_block.pack())