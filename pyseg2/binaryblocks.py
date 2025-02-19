"""
PySeg2
@copyright : Maximilien Lehujeur
2023/05/12
"""

from dataclasses import dataclass, field
from typing import Union, Optional, List

import numpy as np


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
    string_terminator: bytes = b"\x00"
    line_terminator: bytes = b"\n"
    reserved: bytes = b"\x00" * (32 - 14)

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
        string_terminator_length = int.from_bytes(
            buff[8:9],
            byteorder=self.endian,
            signed=False)
        assert string_terminator_length in [1, 2], ValueError(string_terminator_length)
        self.string_terminator = buff[9:9 + string_terminator_length]

        # ==================================
        line_terminator_length = int.from_bytes(
            buff[11:12],
            byteorder=self.endian,
            signed=False)
        assert line_terminator_length in [1, 2],  line_terminator_length
        self.line_terminator = buff[12:12+line_terminator_length]

        # =================
        self.reserved = buff[14:33]
        assert len(self.reserved) == 32 - 14, (len(self.reserved))

    def size(self) -> int:
        """number of bytes according to headers (loading)"""
        return 32

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
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
        assert len(self.string_terminator) in [1, 2], ValueError(len(self.string_terminator))
        buff[8:9] = int.to_bytes(
            len(self.string_terminator),
            length=1,
            byteorder=self.endian,
            signed=False)

        # ==================================
        buff[9:9 + len(self.string_terminator)] = self.string_terminator

        # ==================================
        assert len(self.line_terminator) == len(self.line_terminator)
        assert len(self.line_terminator) in [0, 1, 2], len(self.line_terminator)
        buff[11:12] = int.to_bytes(
            len(self.line_terminator),
            length=1,
            byteorder=self.endian,
            signed=False)

        # =================
        buff[12:12 + len(self.line_terminator)] = self.line_terminator

        # =================
        assert len(self.reserved) == 32 - 14, len(self.reserved)
        buff[14:] = self.reserved

        return bytes(buff)


@dataclass
class TracePointerSubblock:
    """
    The table with the location of each trace in the file
    """

    parent: FileDescriptorSubBlock
    # trace_pointers: np.ndarray =
    #     np.empty([], dtype=np.uint32)  # unsigned int instead of unsigned long??
    trace_pointers: np.ndarray = (
        field(default_factory=lambda: np.empty([], dtype=np.uint32)))

    @property
    def endian(self):
        return self.parent.endian

    @property
    def string_terminator(self):
        return self.parent.string_terminator

    @property
    def number_of_traces(self):
        return self.parent.number_of_traces

    def load(self, fid):
        # to load data I use the value informed in the parent (FileDescriptorSubBlock)
        buff = fid.read(self.parent.size_of_trace_pointer_subblock)
        self.unpack(buff)

    def unpack(self, buff: bytes):
        self.trace_pointers = np.frombuffer(buff, dtype="uint32", count=self.parent.number_of_traces)
        assert len(self.trace_pointers) == self.number_of_traces, (len(self.trace_pointers))

    def size(self) -> int:
        """number of bytes according to headers (loading)"""
        return self.parent.size_of_trace_pointer_subblock

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        return self.trace_pointers.size * self.trace_pointers.dtype.itemsize

    def pack(self) -> bytes:
        # WARNING:
        # packing data must be adjusted to the current state of the parameters
        # Here, I assume that the fields of the parent object are already up-to-date
        nbytes = self.size()
        endian = self.endian
        number_of_traces = self.number_of_traces

        dtype = {"little": "<u4", "big": ">u4"}[endian]

        assert 4 <= nbytes <= 65532
        assert not nbytes % 4
        assert 1 <= self.number_of_traces <= 16383
        assert 1 <= self.number_of_traces <= nbytes // 4

        assert len(self.trace_pointers) == number_of_traces, (len(self.trace_pointers))
        assert self.trace_pointers.dtype == np.dtype("uint32"), self.trace_pointers.dtype

        buff = bytearray(b"\x00" * nbytes)

        buff[:number_of_traces * 4] = \
            self.trace_pointers.astype(dtype).tobytes()

        return bytes(buff)

    @property
    def size_of_free_format_section(self):
        ans = self.trace_pointers[0]  # begining of data section (right after this block)
        # subtract the number of bytes in all blocks before this one
        # WARNING : I trust the current state of the FileDescriptorSubBlock
        ans -= self.size()  # size of this block according to parent
        ans -= self.parent.size()  # size of parent
        return ans


@dataclass
class TraceDescriptorSubBlock:
    """
    The trace header includes a fixed part and a Free format section
    """
    parent: FileDescriptorSubBlock
    identification_bytes: bytes = b"\x22\x44"
    size_of_descriptor_block: int = 0
    size_of_data_block: int = 0
    number_of_samples_in_data_block: int = 0
    data_format_code: bytes = b"\x00"
    reserved: bytes = b"\x00" * (32 - 13)

    @property
    def endian(self):
        return self.parent.endian

    @property
    def string_terminator(self):
        return self.parent.string_terminator

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

        self.size_of_descriptor_block = \
            int.from_bytes(buff[2:4], byteorder=self.endian, signed=False)

        assert self.size_of_descriptor_block % 4 == 0
        assert 32 <= self.size_of_descriptor_block <= 65532, self.size_of_descriptor_block

        self.size_of_data_block = \
            int.from_bytes(buff[4:8], byteorder=self.endian, signed=False)
        self.number_of_samples_in_data_block = \
            int.from_bytes(buff[8:12], byteorder=self.endian, signed=False)
        self.data_format_code = buff[12:13]
        self.reserved = buff[13:]

    def pack(self) -> bytes:
        buff = bytearray(b"\x00" * 32)
        endian = self.endian
        if endian == "little":
            buff[0:2] = b"\x22\x44"
        elif endian == "big":
            buff[2:4] = b"\x44\x22"
        else:
            raise ValueError(endian)

        assert self.size_of_descriptor_block % 4 == 0
        assert 32 <= self.size_of_descriptor_block <= 65532
        buff[2:4] = \
            int.to_bytes(
                self.size_of_descriptor_block,
                length=2,
                byteorder=endian, signed=False)

        buff[4:8] = \
            int.to_bytes(self.size_of_data_block,
                         length=4, byteorder=endian,
                         signed=False)

        buff[8:12] = \
            int.to_bytes(self.number_of_samples_in_data_block,
                         length=4, byteorder=endian,
                         signed=False)
        buff[12:13] = self.data_format_code
        assert len(self.reserved) == 32 - 13
        buff[13:] = self.reserved

        return buff

    def size(self) -> int:
        """number of bytes according to headers (loading)"""
        return 32

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        return 32

    @property
    def size_of_free_format_section(self):
        ans = self.size_of_descriptor_block
        ans -= self.size()  # subtract the bytes in this pre-block
        return ans


@dataclass
class TraceDataBlock:
    """
    The data block
    """
    parent: TraceDescriptorSubBlock
    # data: np.ndarray = np.array([], np.dtype('float32')) NO !!!
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype='float32'))

    @property
    def endian(self):
        return self.parent.endian

    @property
    def number_of_samples_in_data_block(self):
        return self.parent.number_of_samples_in_data_block

    @property
    def data_format_code(self):
        return self.parent.data_format_code

    @property
    def dtype(self):
        bl = {"big": ">", "little": "<"}[self.endian]
        fmt = {b"\x01": "i2",
               b"\x02": "i4",
               b"\x04": "f4",
               b"\x05": "f8"}[self.data_format_code]
        return np.dtype(f'{bl}{fmt}')

    def size(self) -> int:
        """number of bytes according to headers (loading)"""
        return self.dtype.itemsize * self.number_of_samples_in_data_block

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        return self.data.dtype.itemsize * self.data.size

    def load(self, fid):
        buff = fid.read(self.size())
        assert len(buff)
        self.unpack(buff)

    def unpack(self, buff: bytes):
        # print(len(buff), self.number_of_samples_in_data_block, self.dtype, "!ù!ù")
        # print('****', len(buff), self.size(), self.dtype, self.number_of_samples_in_data_block)
        self.data = np.frombuffer(
            buff, dtype=self.dtype,
            count=self.number_of_samples_in_data_block)

        if not self.data.flags.writeable:
            self.data = self.data.copy()

    def pack(self) -> bytes:
        assert self.data.size == self.number_of_samples_in_data_block
        assert self.data.ndim == 1
        assert self.data.dtype.itemsize == self.dtype.itemsize

        # print('?????', self.number_of_bytes(), self.dtype)
        buff = self.data.astype(self.dtype).tobytes()
        return buff


@dataclass
class Seg2String:
    """
    Strings as stored in the Free Format Section of the File header and Trace headers
    """
    parent: Union[TracePointerSubblock, TraceDescriptorSubBlock]
    text: str = ""

    _key: str = ""
    _value: Optional[object] = None

    @property
    def endian(self):
        return self.parent.endian

    @property
    def string_terminator(self):
        return self.parent.string_terminator

    @property
    def offset(self):
        return 2 + len(self.text) + len(self.string_terminator)

    def unpack(self, buff):
        # offset was already read to find the length of buff
        # self.offset = int.from_bytes(buff[:2], byteorder=self.endian, signed=False)

        self.text = buff[2:-len(self.string_terminator)].decode('ascii')

        string_terminator = buff[-len(self.string_terminator):]
        assert string_terminator == self.string_terminator, (string_terminator, self.string_terminator)

    def pack(self) -> bytes:
        # assert self.offset == len(self.text) + 2 + len(self.string_terminator)
        buff = bytearray(b'\x00' * self.offset)

        buff[:2] = int.to_bytes(
            self.offset, length=2,
            byteorder=self.endian,
            signed=False)

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

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        # assert self.offset == len(self.text) + 2 + len(self.string_terminator)
        return self.offset  # len(self.text) + 2 + len(self.string_terminator)

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
    parent: Union[
        TracePointerSubblock,   # for the stream header
        TraceDescriptorSubBlock]   # for the trace headers
    strings: Optional[List[Seg2String]] = None

    @property
    def endian(self):
        return self.parent.endian

    @property
    def string_terminator(self):
        return self.parent.string_terminator

    def size(self) -> int:
        """number of bytes according to headers (loading)"""
        return self.parent.size_of_free_format_section

    def number_of_bytes(self) -> int:
        """actual number of bytes (packing)"""
        assert self.strings is not None
        string: Seg2String
        ans = np.sum([string.number_of_bytes() for string in self.strings])
        ans = int(np.ceil(ans / 4.) * 4.)
        return ans

    def load(self, fid):
        # do not use self.nbytes here
        # loading must follow the header info but packing
        # must follow the current value of the attributes

        # both parent types have this method implemented
        buff = fid.read(self.size())
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

            string = Seg2String(
                parent=self.parent,
                # offset=offset,
                text="")
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
        for n, string in enumerate(self.strings):
            string: Seg2String
            buff += string.pack()

        # padd with zeros to get the right length
        if len(buff) < self.number_of_bytes():
            buff += b"\x00" * (self.number_of_bytes() - len(buff))

        return buff
