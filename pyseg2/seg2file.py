from typing import Union, Optional, List
from dataclasses import dataclass
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

    def pack(self) -> bytes:
        assert 4 <= self.size_of_trace_pointer_subblock <= 65532
        assert not self.size_of_trace_pointer_subblock % 4
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= self.size_of_trace_pointer_subblock // 4

        assert len(self.trace_pointers) == self.number_of_traces, (len(self.trace_pointers))
        assert self.trace_pointers.dtype == np.dtype("uint32")

        buff = bytearray(b"\x00" * self.size_of_trace_pointer_subblock)
        dtype = {"little": ">i4", "big": "<i4"}[self.endian]
        buff[:self.number_of_traces*4] = \
            self.trace_pointers.astype(dtype).tobytes()

        return bytes(buff)


@dataclass
class String:
    """
    Strings as stored in the Free Format Section of the File header and Trace headers
    """
    endian: str = "little"
    offset: int = 0
    text: str = ""
    string_terminator: bytes = b"\x00"
    _key: str = ""
    _value: Optional[object] = None

    def unpack(self, buff):
        # offset was already read to find the length of buff
        # self.offset = int.from_bytes(buff[:2], byteorder=self.endian, signed=False)

        self.text = buff[2:-len(self.string_terminator)].decode('ascii')

        string_terminator = buff[-len(self.string_terminator):]
        assert string_terminator == self.string_terminator

    def pack(self) -> bytes:
        assert self.offset == len(self.text) + 2 + len(self.string_terminator)
        buff = bytearray(b'\x00' * self.offset)
        buff[:2] = int.to_bytes(self.offset, length=2, byteorder=self.endian, signed=False)
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

    def pack(self) -> bytes:
        self.sort()
        buff = b""
        for string in self.strings:
            buff += string.pack()
        return buff


@dataclass
class TraceDescriptorBlock:
    """
    The trace header includes a fixed part and a Free format section
    """
    endian: str = "little"
    string_terminator: bytes = b"\x00"
    identification_bytes: bytes = b"\x22\x44"
    size_of_descriptor_block: int = 0
    size_of_data_block: int = 0
    number_of_samples_in_data_block: int = 0
    data_format_code: bytes = b"\x00"
    reserved: bytes = b"\x00"
    free_format_section: FreeFormatSection = None

    def set(self, trace_pointer_subblock: TracePointerSubblock):
        self.endian = trace_pointer_subblock.endian
        self.string_terminator = trace_pointer_subblock.string_terminator

    def load(self, fid):
        buff = fid.read(32)
        self.trace_descriptor_block_id = buff[0:2]
        if self.endian == "little":
            assert self.trace_descriptor_block_id == b"\x22\x44", self.trace_descriptor_block_id
        else:
            assert self.trace_descriptor_block_id == b"\x44\x22", self.trace_descriptor_block_id
        self.size_of_descriptor_block = int.from_bytes(buff[2:4], byteorder=self.endian)
        assert self.size_of_descriptor_block % 4 == 0
        assert 32 <= self.size_of_descriptor_block <= 65532

        self.size_of_data_block = int.from_bytes(buff[4:8], byteorder=self.endian)
        self.number_of_samples_in_data_block = int.from_bytes(buff[8:12], byteorder=self.endian)
        self.data_format_code = buff[12:13]
        self.reserved = buff[13:]

        self.free_format_section = FreeFormatSection(
            endian=self.endian,
            string_terminator=self.string_terminator,
            size_of_free_format_section=self.size_of_descriptor_block - 32,
            )
        self.free_format_section.load(fid)

    def pack(self) -> bytes:
        raise NotImplementedError



@dataclass
class TraceDataBlock:
    """
    The data block
    """
    data_format_code: bytes = b"\x04"
    number_of_samples_in_data_block: int = 0
    number_of_bytes: int = 0
    dtype: np.dtype = np.dtype('float32')
    data: np.ndarray = np.array([], np.dtype('float32'))

    def set(self, trace_descriptor_block: TraceDescriptorBlock):
        self.data_format_code = trace_descriptor_block.data_format_code
        self.number_of_samples_in_data_block = trace_descriptor_block.number_of_samples_in_data_block

        if self.data_format_code == b"\x03":
            raise NotImplementedError

        self.dtype = {
            b"\x01": np.dtype('int16'),
            b"\x02": np.dtype('int32'),
            b"\x04": np.dtype('float32'),
            b"\x05": np.dtype('float64'),
            }[self.data_format_code]

        self.number_of_bytes = self.number_of_samples_in_data_block * {
            b"\x01": 2,
            b"\x02": 4,
            b"\x04": 4,
            b"\x05": 8,
            }[self.data_format_code]

    def load(self, fid):
        buff = fid.read(self.number_of_bytes)
        self.unpack(buff)

    def unpack(self, buff: bytes):
        self.data = np.frombuffer(buff, dtype=self.dtype, count=self.number_of_samples_in_data_block)

    def pack(self) -> bytes:
        raise NotImplementedError

@dataclass
class Seg2Trace:
    trace_descriptor_block: TraceDescriptorBlock
    trace_data_block: TraceDataBlock


class Seg2File:
    def __init__(self, filename: str):

        self.file_descriptor_subblock = FileDescriptorSubBlock()
        self.trace_pointer_subblock = TracePointerSubblock()
        self.free_format_section = FreeFormatSection()
        self.seg2traces = []

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


if __name__ == "__main__":

    seg2 = Seg2File('./toto.seg2')
    # print(seg2.file_descriptor_subblock)
    # print(seg2.file_descriptor_subblock.pack())
    #
    # print(seg2.trace_pointer_subblock)
    # print(seg2.trace_pointer_subblock.pack())

    # print(seg2.free_format_section)
    # print(seg2.free_format_section.pack())
