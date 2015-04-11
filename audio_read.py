"""Reads in an audio file with resampling, all at once or in chunks.

audio_read() reads in a whole soundfile at once (or chosen subregion).

e.g.

  d, sr = audio_read.audio_read(filename, sr=desired_sr,
                                channels=desired_chans,
                                offset=desired_start_time_sec,
                                duration=desired_max_duration_sec)
  # Returns d as an np.array (num_frames, num_channels)

The AudioReader object allows you to incrementally walk through a
sound file (useful for very long files).

e.g.

  reader = audio_read.AudioReader(filename, sr=sr, channels=1)
  chunk_duration_seconds = 10.0
  while not reader.done():
    # Read the next chunk of samples.
    d, sr = reader.read(chunk_duration_seconds)
    # Process samples in d.
    ...

"""

# Equivalent to:
#import librosa
#def audio_read(filename, sr=11025, channels=1):
#    """Read a soundfile, return (d, sr)."""
#    d, sr = librosa.load(filename, sr=sr, mono=(channels == 1))
#    return d, sr

# The code below is adapted from:
# https://github.com/bmcfee/librosa/blob/master/librosa/core/audio.py
# This is its original copyright notice:

# Copyright (c) 2014, Brian McFee, Matt McVicar, Dawen Liang, Colin Raffel, Douglas Repetto, Dan Ellis.
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os
import numpy as np

def audio_read(filename, sr=None, channels=None, offset=None, duration=None):
    """Read a soundfile, return (d, sr)."""
    ao = AudioReader(filename, sr, channels, offset, duration)
    return ao.read()

class AudioReader(object):
    """Read a soundfile, return (d, sr)."""
    # Hacked version of librosa.load and audioread/ff.
    def __init__(self, filename=None, sr=None, channels=None, offset=None, max_duration=None):
        self.filename = filename
        self.sr = sr
        self.channels = channels
        self.offset = offset
        self.max_duration = max_duration
        if self.filename:
            self.open(self.filename)

    def open(self, filename, sr=None, channels=None, offset=None, max_duration=None):
        if sr:
            self.sr = sr
        if channels:
            self.channels = channels
        if offset:
            self.offset = offset
        if not self.offset:
            self.offset = 0.0
        if max_duration:
            self.max_duration = max_duration
        self.filename = filename
        self.input_file = FFmpegAudioFile(os.path.realpath(filename),
                                          sample_rate=self.sr, channels=self.channels)
        self.input_file_iterator = self.input_file.read_data()
        self.sr = self.input_file.sample_rate
        self.channels = self.input_file.channels
        # ffmpeg also reports the duration if possible.
        self.duration = self.input_file.duration
        self.start_sample = int(np.floor(self.sr * self.offset) * self.channels)
        if self.max_duration:
            self.last_sample = int(np.ceil(self.sr * self.max_duration) * self.channels)
        else:
            self.last_sample = None
        self.saved_samples = np.zeros((0,))
        self.sample_position = 0
        self.eof = False

    def done(self):
        return self.eof

    def time(self):
        # Return the current read position within the soundfile in seconds.
        return float(self.sample_position)/(self.channels * self.sr)

    def read(self, duration=None):
        dtype = np.float32
        if duration is None:
            num_to_read = np.inf
        else:
            num_to_read = int(np.ceil(self.sr * duration) * self.channels)
        # Enforce any max_duration
        if self.last_sample:
            num_to_read = min(num_to_read, self.last_sample - self.sample_position)
        y = [self.saved_samples]
        num_read = self.saved_samples.shape[0]
        while num_read < num_to_read and not self.eof:
            try:
                frame = buf_to_float(self.input_file_iterator.next(), dtype=dtype)
            except StopIteration:
                self.eof = True
                break
            num_this_frame = len(frame)
            self.sample_position += num_this_frame
            if self.sample_position > self.start_sample:
                # First frame with valid samples - discard any before self.start_sample.
                num_this_frame = min(num_this_frame, self.sample_position - self.start_sample)
                frame = frame[-num_this_frame:]
                # tack on the current frame
                y.append(frame)
                num_read += num_this_frame
            if num_read >= num_to_read:
                break

        if not len(y):
            # Zero-length read
            y = np.zeros(0, dtype=dtype)
        else:
            y = np.concatenate(y)

        num_to_read = min(num_to_read, y.shape[0])

        # Buffer any excess
        self.saved_samples = np.copy(y[num_to_read:])
        y = y[:num_to_read]

        if self.channels > 1:
            y = y.reshape((-1, 2)).T

        # Final cleanup for dtype and contiguity
        y = np.ascontiguousarray(y, dtype=dtype)

        return (y, self.sr)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    .. seealso:: :func:`librosa.util.buf_to_float`
    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer
        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``
        - dtype : numeric type
            The target output type (default: 32-bit float)
    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


# The code below is adapted from:
# https://github.com/sampsyo/audioread/blob/master/audioread/ffdec.py
# Below is its original copyright notice:

# This file is part of audioread.
# Copyright 2014, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.


import subprocess
import re
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue


class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """
    def __init__(self, fh, blocksize=1024, discard=False):
        super(QueueReaderThread, self).__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                # Stream closed (EOF).
                break


class FFmpegAudioFile(object):
    """An audio file decoded by the ffmpeg command-line utility."""
    def __init__(self, filename, channels=None, sample_rate=None, block_size=4096):
        popen_args = ['ffmpeg', '-i', filename, '-f', 's16le']
        self.channels = channels
        self.sample_rate = sample_rate
        if channels:
            popen_args.extend(['-ac', str(channels)])
        if sample_rate:
            popen_args.extend(['-ar', str(sample_rate)])
        popen_args.append('-')
        self.proc = subprocess.Popen(
            popen_args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        self._get_info()

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ValueError('ffmpeg output: {}'.format(
                            ''.join(self.stderr_reader.queue.queue)
                        ))
                    else:
                        start_time = end_time
                        # Keep waiting.
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise ValueError("stream info not found")

            # In Python 3, result of reading from stderr is bytes.
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')

            line = line.strip().lower()

            if 'no such file' in line:
                raise IOError('file not found')
            elif 'invalid data found' in line:
                raise UnsupportedError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        # Sample rate.
        match = re.search(r'(\d+) hz', s)
        if match:
            self.sample_rate_orig = int(match.group(1))
        else:
            self.sample_rate_orig = 0
        if self.sample_rate is None:
            self.sample_rate = self.sample_rate_orig

        # Channel count.
        match = re.search(r'hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels_orig = 2
            else:
                match = re.match(r'(\d+) ', mode)
                if match:
                    self.channels_orig = int(match.group(1))
                else:
                    self.channels_orig = 1
        else:
            self.channels_orig = 0
        if self.channels is None:
            self.channels = self.channels_orig

        # Duration.
        match = re.search(
            r'duration: (\d+):(\d+):(\d+).(\d)', s
        )
        if match:
            durparts = list(map(int, match.groups()))
            duration = (
                durparts[0] * 60 * 60 +
                durparts[1] * 60 +
                durparts[2] +
                float(durparts[3]) / 10
            )
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        # Kill the process if it is still running.
        if hasattr(self, 'proc') and self.proc.returncode is None:
            self.proc.kill()
            self.proc.wait()

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_data()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
