from dataclasses import dataclass
import numpy as np


@dataclass
class FileDescriptorSubBlock:
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
        if self.revision_number == 1:
            pass
        else:
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
        raise NotImplementedError


@dataclass
class TracePointerSubblock:

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
        print(self.size_of_trace_pointer_subblock, len(buff), self.number_of_traces)
        self.unpack(buff)

    def unpack(self, buff: bytes):
        self.trace_pointers = np.frombuffer(buff, dtype="uint32", count=self.number_of_traces)
        assert len(self.trace_pointers) == self.number_of_traces, (len(self.trace_pointers))


@dataclass
class String:
    offset: int = 0
    text: str = ""
    string_terminator: bytes = b"\x00"

    def __str__(self):
        return f"{self.offset=} {self.text=} {self.string_terminator=}"


@dataclass
class FreeFormatSection:

    endian: str = "little"
    string_terminator: bytes = b"\x00"
    size_of_free_format_section: int = 0
    strings: np.ndarray = np.empty([], object)

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

            stringbuff = buff[i+2:i+offset]
            text = stringbuff[:-len(self.string_terminator)].decode('ascii')
            tailer = stringbuff[-len(self.string_terminator):]
            assert tailer == self.string_terminator

            string = String(
                offset=offset,
                text=text,
                string_terminator=tailer)
            print(string.text)

            if text.startswith('NOTE'):
                break
            i += offset


@dataclass
class TraceDescriptorBlock:
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


@dataclass
class DataBlock:
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

        self.number_of_bytes =  self.number_of_samples_in_data_block * {
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


if __name__ == "__main__":

    fdsb = FileDescriptorSubBlock()
    tpsb = TracePointerSubblock()
    ffs = FreeFormatSection()

    with open('toto.seg2', 'rb') as fid:
        fdsb.load(fid)
        print(fdsb)

        tpsb.set(fdsb)
        tpsb.load(fid)
        print(tpsb)

        ffs.set(fdsb, tpsb)
        ffs.load(fid)

        for n, i in enumerate(tpsb.trace_pointers):
            fid.seek(i, 0)
            tdb = TraceDescriptorBlock()
            db = DataBlock()

            tdb.set(tpsb)
            tdb.load(fid)
            print(tdb)

            db.set(tdb)
            db.load(fid)

            print(db)
            import matplotlib.pyplot as plt
            plt.plot(0.1 * db.data / np.std(db.data) + n)
    plt.show()