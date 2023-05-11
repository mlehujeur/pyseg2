from typing import Union, Optional, List
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# TODO : implement the pack method of each block for writing


@dataclass
class FileDescriptorSubBlock:
    """
    First part of the File Descriptor
    """
    endian: str = "little"
    identification_bytes: bytes = b"\x55\x3a"
    revision_number: int = 1
    size_of_trace_pointer_subblock: int = 4   # M
    number_of_traces: int = 1  # N
    string_terminator_length: int = 1
    string_terminator: bytes = b"\x00"
    line_terminator_length: int = 1
    line_terminator: str = "\n"
    reserved: bytes = b"\x00" * (32 - 14 + 1)

    def load(self, fid) -> None:
        buff = fid.read(32)
        # self._buffin = buff
        self.unpack(buff)

    def unpack(self, buff: bytes) -> None:
        # ==================================
        self.identification_bytes = buff[0:2]

        if self.identification_bytes == b"\x55\x3a":
            self.endian = "little"
        elif self.identification_bytes == b"\x3a\x55":
            self.endian = "big"
        else:
            raise IOError("not a seg2 file, {self.identification_bytes}")

        # ==================================
        self.revision_number = int.from_bytes(
            buff[2:4], 
            byteorder=self.endian, 
            signed=False)
        if self.revision_number != 1:
            raise NotImplementedError(self.revision_number)

        # ==================================        
        self.size_of_trace_pointer_subblock = int.from_bytes(
            buff[4:6], 
            byteorder=self.endian, 
            signed=False)
        assert 4 <= self.size_of_trace_pointer_subblock <= 65532
        assert not self.size_of_trace_pointer_subblock % 4 

        # ==================================
        self.number_of_traces = int.from_bytes(
            buff[6:8], 
            byteorder=self.endian, 
            signed=False)
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= self.size_of_trace_pointer_subblock // 4

        # ==================================
        self.string_terminator_length = int.from_bytes(
            buff[8:9],
            byteorder=self.endian,
            signed=False)
        assert self.string_terminator_length in [1, 2], ValueError(self.string_terminator_length)

        # =================
        self.string_terminator = buff[9:9 + self.string_terminator_length]

        # ==================================
        self.line_terminator_length = int.from_bytes(
            buff[11:12],
            byteorder=self.endian,
            signed=False)
        assert self.line_terminator_length in [0, 1, 2],  self.line_terminator_length

        # =================
        if self.line_terminator_length == 0:
            self.line_terminator = ""

        elif self.line_terminator_length == 1:
            self.line_terminator = buff[12:13]

        elif self.line_terminator_length == 2:
            self.line_terminator = buff[12:14]
        
        else:
            raise ValueError(self.line_terminator_length)

        # =================
        self.reserved = buff[14:33]
        assert len(self.reserved) == 32 - 14, (len(self.reserved))

    def nbytes(self):
        """
        number of bytes in this block
        """
        return 32

    def pack(self) -> bytes:
        buff = bytearray(b"\x00" * 32)

        # =================
        if self.endian == "little":
            buff[0:2] = b"\x55\x3a"
        elif self.endian == "big":
            buff[0:2] = b"\x3a\x55"
        else:
            raise ValueError(self.endian)

        # =================
        if self.revision_number != 1:
            raise NotImplementedError(self.revision_number)

        buff[2:4] = int.to_bytes(
            self.revision_number, length=2,
            byteorder=self.endian, signed=False)

        # =================
        assert 4 <= self.size_of_trace_pointer_subblock <= 65532
        assert not self.size_of_trace_pointer_subblock % 4
        # WARNING : size must match with the trace pointer subblock
        buff[4:6] = int.to_bytes(
            self.size_of_trace_pointer_subblock,
            length=2,
            byteorder=self.endian,
            signed=False)

        # =================
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= self.size_of_trace_pointer_subblock // 4
        buff[6:8] = int.to_bytes(
            self.number_of_traces,
            length=2,
            byteorder=self.endian,
            signed=False)

        # ==================================
        assert self.string_terminator_length in [1, 2], ValueError(self.string_terminator_length)
        assert len(self.string_terminator) == self.string_terminator_length
        buff[8:9] = int.to_bytes(
            self.string_terminator_length,
            length=1,
            byteorder=self.endian,
            signed=False)

        # ==================================
        buff[9:9 + self.string_terminator_length] = self.string_terminator

        # ==================================
        assert len(self.line_terminator) == self.line_terminator_length
        assert self.line_terminator_length in [0, 1, 2], self.line_terminator_length
        buff[11:12] = int.to_bytes(
            self.line_terminator_length,
            length=1,
            byteorder=self.endian,
            signed=False)

        # =================
        buff[12:12 + self.line_terminator_length] = self.line_terminator

        # =================
        assert len(self.reserved) == 32 - 14
        buff[14:] = self.reserved

        # print(self._buffin)
        # print(bytes(buff))
        # raise
        return bytes(buff)


@dataclass
class TracePointerSubblock:
    """
    The table with the location of each trace in the file
    """
    endian: str = "little"
    string_terminator: bytes = b"\x00"
    number_of_traces: int = 1  # N
    size_of_trace_pointer_subblock: int = 4   # M
    trace_pointers: np.ndarray = \
        np.empty(size_of_trace_pointer_subblock, dtype=np.uint32)  # unsigned int instead of unsigned long??

    def set(self, file_descriptor_subblock: FileDescriptorSubBlock):
        self.endian = file_descriptor_subblock.endian
        self.number_of_traces = file_descriptor_subblock.number_of_traces
        self.size_of_trace_pointer_subblock = file_descriptor_subblock.size_of_trace_pointer_subblock
        self.string_terminator = file_descriptor_subblock.string_terminator

        assert 4 <= self.size_of_trace_pointer_subblock <= 65532
        assert not self.size_of_trace_pointer_subblock % 4
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= self.size_of_trace_pointer_subblock // 4

    def load(self, fid):
        buff = fid.read(self.size_of_trace_pointer_subblock)
        self.unpack(buff)

    def unpack(self, buff: bytes):
        self.trace_pointers = np.frombuffer(buff, dtype="uint32", count=self.number_of_traces)
        assert len(self.trace_pointers) == self.number_of_traces, (len(self.trace_pointers))

    def nbytes(self):
        return self.size_of_trace_pointer_subblock

    def pack(self) -> bytes:
        assert 4 <= self.size_of_trace_pointer_subblock <= 65532
        assert not self.size_of_trace_pointer_subblock % 4
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= self.size_of_trace_pointer_subblock // 4

        assert len(self.trace_pointers) == self.number_of_traces, (len(self.trace_pointers))
        assert self.trace_pointers.dtype == np.dtype("uint32"), self.trace_pointers.dtype

        buff = bytearray(b"\x00" * self.size_of_trace_pointer_subblock)
        dtype = {"little": "<u4", "big": ">u4"}[self.endian]

        buff[:self.number_of_traces*4] = \
            self.trace_pointers.astype(dtype).tobytes()

        return bytes(buff)


@dataclass
class String:
    """
    Strings as stored in the Free Format Section of the File header and Trace headers
    """
    endian: str = "little"
    offset: int = 0   # replace by a property?
    text: str = ""
    string_terminator: bytes = b"\x00"
    _key: str = ""
    _value: Optional[object] = None

    def unpack(self, buff):
        # offset was already read to find the length of buff
        # self.offset = int.from_bytes(buff[:2], byteorder=self.endian, signed=False)

        self.text = buff[2:-len(self.string_terminator)].decode('ascii')

        string_terminator = buff[-len(self.string_terminator):]
        assert string_terminator == self.string_terminator, (string_terminator, self.string_terminator)

    def pack(self) -> bytes:
        assert self.offset == len(self.text) + 2 + len(self.string_terminator)
        buff = bytearray(b'\x00' * self.offset)

        buff[:2] = int.to_bytes(
            self.offset, length=2,
            byteorder=self.endian, signed=False)

        buff[2:-len(self.string_terminator)] = self.text.encode('ascii')
        buff[-len(self.string_terminator):] = self.string_terminator

        return buff

    def extract(self):
        text = self.text.strip()
        key = text.split()[0].split('\t')[0]
        value = text.split(key)[-1].strip()
        if "\n" in value:
            value = value.split('\n')
            value = [_.strip() for _ in value]
            while "" in value:
                value.remove("")
            value = ";".join(value)
        return key, value

    def nbytes(self):
        # return self.offset # presumed number of nytes
        assert self.offset == len(self.text) + 2 + len(self.string_terminator)
        return len(self.text) + 2 + len(self.string_terminator)

    @property
    def key(self):
        if self._key == "":
            self._key, self._value = self.extract()
        return self._key

    @property
    def value(self):
        if self._value is None:
            self._key, self._value = self.extract()
        return self._value

    def __str__(self):
        return f"STRING {self.key}: {self.value}"

@dataclass
class FreeFormatSection:
    """
    A group of strings that appear in the file header and in each trace header
    """
    endian: str = "little"
    string_terminator: bytes = b"\x00"
    size_of_free_format_section: int = 0
    strings: Optional[List[str]] = None

    def set(self,
            file_descriptor_subblock: FileDescriptorSubBlock,
            trace_pointer_subblock: TracePointerSubblock):

        self.endian = file_descriptor_subblock.endian
        self.string_terminator = file_descriptor_subblock.string_terminator
        self.size_of_free_format_section = \
            trace_pointer_subblock.trace_pointers[0] \
            - 32 \
            - trace_pointer_subblock.size_of_trace_pointer_subblock

    def set_endian(self, endian: str):
        self.endian = endian
        for string in self.strings:
            string.endian = endian

    def load(self, fid):
        buff = fid.read(self.size_of_free_format_section)
        self.unpack(buff=buff)

    def unpack(self, buff: bytes):
        i = 0
        offset = 1
        while offset:
            offset = int.from_bytes(
                buff[i:i+2],
                byteorder=self.endian, signed=False)

            if offset == 0:
                # An offset of 0 (2 bytes),
                # after the final string, marks the end of the
                # string list
                break

            string = String(
                offset=offset,
                text="",
                string_terminator=self.string_terminator)
            string.unpack(buff[i: i+offset])

            if self.strings is None:
                self.strings = [string]
            else:
                self.strings.append(string)

            if string.text.startswith('NOTE'):
                # NOTE is supposed to be the last keyword if present
                break
            i += offset

    def sort(self):
        """keys must be listed in alphabetical order, NOTE must be the last one"""
        keys = [string.key for string in self.strings]

        note = None
        if "NOTE" in keys:
            # extract note from self.strings
            i = keys.index('NOTE')
            note = self.strings.pop(i)
            keys.pop(i)

        # sort by alphabetical order
        self.strings = [self.strings[j] for j in np.argsort(keys)]

        # put note at the end of the list
        if note is not None:
            self.strings.append(note)

    def nbytes(self):
        #return sum([string.nbytes() for string in self.strings])  # might be shorter than actual length
        return self.size_of_free_format_section

    def pack(self) -> bytes:
        self.sort()
        buff = b""
        for n, string in enumerate(self.strings):
            string: String
            buff += string.pack()

        # padd with zeros to get the right length
        if len(buff) < self.size_of_free_format_section:
            buff += b"\x00" * (self.size_of_free_format_section - len(buff))

        return buff


@dataclass
class TraceDescriptorBlock:
    """
    The trace header includes a fixed part and a Free format section
    """
    endian: str = "little"
    string_terminator: bytes = b"\x00"
    identification_bytes: bytes = b"\x22\x44"
    size_of_data_block: int = 0
    number_of_samples_in_data_block: int = 0
    data_format_code: bytes = b"\x00"
    reserved: bytes = b"\x00"
    free_format_section: FreeFormatSection = None

    def set(self, trace_pointer_subblock: TracePointerSubblock):
        self.endian = trace_pointer_subblock.endian
        self.string_terminator = \
            trace_pointer_subblock.string_terminator

    def load(self, fid):
        buff = fid.read(32)
        assert len(buff) == 32

        self.trace_descriptor_block_id = buff[0:2]
        if self.endian == "little":
            assert self.trace_descriptor_block_id == b"\x22\x44", \
                self.trace_descriptor_block_id
        else:
            assert self.trace_descriptor_block_id == b"\x44\x22", \
                self.trace_descriptor_block_id

        size_of_descriptor_block = \
            int.from_bytes(buff[2:4], byteorder=self.endian, signed=False)

        assert size_of_descriptor_block % 4 == 0
        assert 32 <= size_of_descriptor_block <= 65532, size_of_descriptor_block

        self.size_of_data_block = \
            int.from_bytes(buff[4:8], byteorder=self.endian, signed=False)
        self.number_of_samples_in_data_block = \
            int.from_bytes(buff[8:12], byteorder=self.endian, signed=False)
        self.data_format_code = buff[12:13]
        self.reserved = buff[13:]

        self.free_format_section = FreeFormatSection(
            endian=self.endian,
            string_terminator=self.string_terminator,
            size_of_free_format_section=size_of_descriptor_block - 32,
            )
        self.free_format_section.load(fid)

    def pack(self) -> bytes:
        buff = bytearray(b"\x00" * 32)

        if self.endian == "little":
            buff[0:2] = b"\x22\x44"
        elif self.endian == "big":
            buff[2:4] = b"\x44\x22"
        else:
            raise ValueError(self.endian)

        size_of_descriptor_block = self.nbytes()
        assert size_of_descriptor_block % 4 == 0
        assert 32 <= size_of_descriptor_block <= 65532
        buff[2:4] = \
            int.to_bytes(
                size_of_descriptor_block,
                length=2,
                byteorder=self.endian, signed=False)

        buff[4:8] = \
            int.to_bytes(self.size_of_data_block,
                         length=4, byteorder=self.endian,
                         signed=False)

        buff[8:12] = \
            int.to_bytes(self.number_of_samples_in_data_block,
                         length=4, byteorder=self.endian,
                         signed=False)
        buff[12:13] = self.data_format_code
        assert len(self.reserved) == 32 - 13
        buff[13:] = self.reserved

        buff += self.free_format_section.pack()
        return buff

    def nbytes(self):
        return 32 + self.free_format_section.nbytes()

    # @property
    # def size_of_descriptor_block(self):
    #     return self.nbytes()

@dataclass
class TraceDataBlock:
    """
    The data block
    """
    endian: str = "little"
    data_format_code: bytes = b"\x04"
    number_of_samples_in_data_block: int = 0
    data: np.ndarray = np.array([], np.dtype('float32'))

    @property
    def dtype(self):
        bl = {"big": ">", "little": "<"}[self.endian]
        fmt = {b"\x01": "i2",
               b"\x02": "i4",
               b"\x04": "f4",
               b"\x05": "f8"}[self.data_format_code]
        return np.dtype(f'{bl}{fmt}')

    def nbytes(self):
        return self.dtype.itemsize * self.number_of_samples_in_data_block

    def set(self, trace_descriptor_block: TraceDescriptorBlock):
        self.endian = trace_descriptor_block.endian
        self.data_format_code = trace_descriptor_block.data_format_code
        self.number_of_samples_in_data_block = trace_descriptor_block.number_of_samples_in_data_block

        if self.data_format_code == b"\x03":
            raise NotImplementedError

    def load(self, fid):
        buff = fid.read(self.nbytes())
        self.unpack(buff)

    def unpack(self, buff: bytes):
        # print(len(buff), self.number_of_samples_in_data_block, self.dtype, "!ù!ù")

        self.data = np.frombuffer(
            buff, dtype=self.dtype,
            count=self.number_of_samples_in_data_block)

    def pack(self) -> bytes:
        assert len(self.data) == self.number_of_samples_in_data_block

        buff = self.data.astype(self.dtype).tobytes()
        return buff


@dataclass
class Seg2Trace:
    trace_descriptor_block: TraceDescriptorBlock
    trace_data_block: TraceDataBlock

    def set_endian(self, endian: str):
        self.trace_descriptor_block.endian = endian
        self.trace_data_block.endian = endian

    def pack(self):
        return self.trace_descriptor_block.pack() + \
            self.trace_data_block.pack()

    def nbytes(self):
        return self.trace_descriptor_block.nbytes() + \
            self.trace_data_block.nbytes()

class Seg2File:
    def __init__(self, filename: str):

        self.file_descriptor_subblock = FileDescriptorSubBlock()
        self.trace_pointer_subblock = TracePointerSubblock()
        self.free_format_section = FreeFormatSection()
        self.seg2traces: List[Seg2Trace] = []

        with open(filename, 'rb') as fid:
            self.file_descriptor_subblock.load(fid)

            self.trace_pointer_subblock.set(self.file_descriptor_subblock)
            self.trace_pointer_subblock.load(fid)

            self.free_format_section.set(
                self.file_descriptor_subblock,
                self.trace_pointer_subblock)
            self.free_format_section.load(fid)

            for n, trace_pointer in enumerate(self.trace_pointer_subblock.trace_pointers):
                # make sure the cursor is positioned at the beginning
                # of the trace
                fid.seek(trace_pointer, 0)

                trace_descriptor_block = TraceDescriptorBlock()
                trace_data_block = TraceDataBlock()

                trace_descriptor_block.set(self.trace_pointer_subblock)
                trace_descriptor_block.load(fid)

                trace_data_block.set(trace_descriptor_block)
                trace_data_block.load(fid)

                seg2trace = Seg2Trace(
                    trace_descriptor_block,
                    trace_data_block)

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

    def set_endian(self, endian: str):
        assert endian in ["little", "big"]
        self.file_descriptor_subblock.endian = endian
        self.trace_pointer_subblock.endian = endian
        self.free_format_section.set_endian(endian)
        for trace in self.seg2traces:
            trace.set_endian(endian)

    def pack(self) -> bytes:

        # homogenise the endianess
        self.set_endian(self.file_descriptor_subblock.endian)

        # ==== recompute the trace pointer table
        number_of_traces = len(self.seg2traces)
        self.file_descriptor_subblock.number_of_traces = number_of_traces

        # reset the trace pointer table
        self.trace_pointer_subblock.number_of_traces = number_of_traces
        self.trace_pointer_subblock.size_of_trace_pointer_subblock = \
            4 * number_of_traces
        self.trace_pointer_subblock.trace_pointers = \
            np.empty(self.trace_pointer_subblock.number_of_traces,
                     dtype=np.uint32)

        # length of the file descriptor block
        nbytes = \
            self.file_descriptor_subblock.nbytes() + \
            self.trace_pointer_subblock.nbytes() + \
            self.free_format_section.nbytes()

        self.trace_pointer_subblock.trace_pointers[0] = nbytes  # number of bytes in the header
        for n, trace in enumerate(self.seg2traces[:-1]):
            # put the number of bytes of the trace for now
            # x = self.trace_pointer_subblock.trace_pointers.dtype.type(trace.nbytes())
            self.trace_pointer_subblock.trace_pointers[n+1] = trace.nbytes()

        # convert number of bytes into positions
        self.trace_pointer_subblock.trace_pointers = \
            self.trace_pointer_subblock.trace_pointers\
                .cumsum()\
                .astype('uint32')
        # print(self.trace_pointer_subblock.trace_pointers)
        # self.trace_pointer_subblock.unpack(self.trace_pointer_subblock.pack())
        # print(self.trace_pointer_subblock.trace_pointers)

        buff = \
            self.file_descriptor_subblock.pack() + \
            self.trace_pointer_subblock.pack() + \
            self.free_format_section.pack()
        assert len(buff) == nbytes

        for n, trace in enumerate(self.seg2traces):
            # the current position must agree with the trace_pointer_table
            assert len(buff) == self.trace_pointer_subblock.trace_pointers[n]

            buff += trace.pack()
            # print("packing", n,
            #       trace.trace_descriptor_block.nbytes(),
            #       len(trace.trace_descriptor_block.pack()),
            #       len(trace.trace_data_block.pack()))

        return buff


if __name__ == "__main__":

    print('load toto')
    seg2 = Seg2File('./toto.seg2')

    print('write tata')
    with open('tata.seg2', 'wb') as fil:
        fil.write(seg2.pack())

    print('load tata')
    seg2re = Seg2File('./tata.seg2')
    print(seg2re.seg2traces[0].trace_descriptor_block.free_format_section)

    exit()

    print('load tata')
    with open('tata.seg2', 'rb') as fid:
        file_descriptor_subblock = FileDescriptorSubBlock()
        file_descriptor_subblock.load(fid)

        trace_pointer_subblock = TracePointerSubblock()
        trace_pointer_subblock.set(file_descriptor_subblock)
        trace_pointer_subblock.load(fid)

        free_format_section = FreeFormatSection()
        free_format_section.set(file_descriptor_subblock, trace_pointer_subblock)
        free_format_section.load(fid)

        # fid.seek(trace_pointer_subblock.trace_pointers[0], 0)
        trace_descriptor_block = TraceDescriptorBlock()
        trace_descriptor_block.set(trace_pointer_subblock)
        trace_descriptor_block.load(fid)


        trace_data_block = TraceDataBlock()
        trace_data_block.set(trace_descriptor_block)
        print(fid.tell())
        trace_data_block.load(fid)
        print(fid.tell())



    # print(seg2.file_descriptor_subblock)
    # print(file_descriptor_subblock)

    # print(seg2.trace_pointer_subblock)
    # print(trace_pointer_subblock)
    # print(np.abs(seg2.trace_pointer_subblock.trace_pointers - trace_pointer_subblock.trace_pointers).sum())

    # for string in seg2.free_format_section.strings:
    #     print(string)
    # print()
    # for string in free_format_section.strings:
    #     print(string)

    # print(str(seg2.seg2traces[0].trace_descriptor_block).replace(',', ',\n\t'))
    # print(str(trace_descriptor_block).replace(',', ',\n\t'))
    #
    # print(seg2.seg2traces[0].trace_data_block.data.shape)
    # print(trace_data_block.data.shape)
    plt.figure()
    plt.plot(seg2.seg2traces[0].trace_data_block.data)
    plt.plot(trace_data_block.data)
    plt.show()

    # seg2re = Seg2File('./tata.seg2')
    # print(seg2re.seg2traces[0].trace_descriptor_block.free_format_section)






    # print(seg2.file_descriptor_subblock)
    # print(seg2.file_descriptor_subblock.pack())
    #
    # print(seg2.trace_pointer_subblock)
    # print(seg2.trace_pointer_subblock.pack())

    # print(seg2.free_format_section)
    # print(seg2.free_format_section.pack())

    # print(seg2.seg2traces[0].trace_descriptor_block.pack())
    # print(seg2.seg2traces[0].trace_data_block.pack())