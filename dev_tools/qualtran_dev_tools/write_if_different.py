#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import io
from pathlib import Path


class WriteIfDifferent:
    """A file-like object that only writes to disk if the new content
    differs from the existing content.

    Args:
        path: The path to write, which may already exist.
    """

    def __init__(self, path: Path):
        self.path = path
        self._buffer = io.StringIO()

    def write(self, s: str):
        return self._buffer.write(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._buffer.flush()

    def close(self):
        """Closes the adapter.

        This triggers the comparison of buffered content
        with the disk file's content and writes to disk only if different.
        """
        new_content = self._buffer.getvalue()
        self._buffer.close()

        existing_content = None
        if self.path.is_file():
            with self.path.open('r') as f_read:
                existing_content = f_read.read()
            if new_content == existing_content:
                print(f"{self.path}\t unchanged.")
                return

        with self.path.open('w') as f_write:
            print(f"{self.path}\t writing.")
            f_write.write(new_content)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Do not suppress exceptions from the 'with' block body.
        return False

    @property
    def closed(self):
        return self._buffer.closed

    def readable(self):
        """Returns False, as this adapter is write-only like a file from `open('w')`."""
        return False

    def writable(self):
        """Returns True if the adapter is not closed, False otherwise."""
        return self._buffer.writable()

    def seekable(self):
        """Returns False, as this adapter is not seekable like a disk file opened in 'w' mode."""
        return False

    def tell(self):
        """Returns the current stream position in the internal buffer."""
        return self._buffer.tell()

    def truncate(self, size=None):
        """
        Resizes the internal buffer to the given number of bytes.
        If size is not specified, resizes to the current position.
        """
        return self._buffer.truncate(size)
