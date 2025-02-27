import sys

from datetime import datetime
from typing import Optional


class OutputLogger(object):
    def __init__(self) -> None:
        self.file = None
        self.mode = None
        self.buffer = ""

    def set_log_file(self, filename: str, mode: str = "at") -> None:
        self.file = filename
        self.mode = mode

        if self.buffer is not None:
            with open(self.file, self.mode) as f:
                f.write(self.buffer)
                self.buffer = None

    def write(self, data: str) -> None:
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self) -> None:
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.flush()


class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush
        self.buffer = ""

    def write(self, data) -> None:
        data = self.processString(data)

        if data is not None:
            for stream in self.child_streams:
                stream.write(data)
            if self.autoflush:
                self.flush()

    def flush(self) -> None:
        for stream in self.child_streams:
            stream.flush()

    def processString(self, string: str) -> str:
        string = string.replace("\n", "\n                          ")

        index = self.findLastBreakLine(string)

        if index is not None:
            string = string[: index + 1]

        return string

    def findLastBreakLine(self, string) -> Optional[int]:
        for i in range(1, len(string) + 1):
            if string[-i] == " ":
                continue
            elif string[-i] == "\n":
                return -i
            else:
                return None


output_logger = None


def init_logs(file: str, mode="at") -> None:
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        builtin_print(datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-3], ":", *args, **kwargs)

    __builtin__.print = print

    global output_logger

    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)
        output_logger.set_log_file(file, mode)
