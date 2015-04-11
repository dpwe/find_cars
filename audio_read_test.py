"""Tests for audio_read."""

import sys

import audio_read
import numpy as np

test_filename = "/Users/dpwe/Downloads/karongwe/2014-05-25 03-Test_40min.wma"

short_test_filename = "tmp.wav"

import unittest

class TestAudioRead(unittest.TestCase):

  def _testReadInParts(self, filename, sr=None, chans=None):
      reader = audio_read.AudioReader(filename, sr=sr, channels=chans)
      chunk_time = 10.0
      whole_thing = reader.read(2*chunk_time)[0]
      reader = audio_read.AudioReader(filename, sr=sr, channels=chans)
      first_chunk = reader.read(chunk_time)[0]
      next_chunk = reader.read(chunk_time)[0]
      np.testing.assert_equal(whole_thing, np.hstack([first_chunk, next_chunk]))

  def _testReadWhole(self, filename, sr=None, chans=None):
      reader = audio_read.AudioReader(filename, sr=sr, channels=chans)
      d, sr = reader.read()
      print len(d)
      print len(d)/float(sr)
      print reader.duration
      # Duration returned from ffmpeg and stored ir reader.duration is only
      # good to one decimal place.
      self.assertAlmostEqual(d.shape[-1]/float(sr), reader.duration, places=1)

  def testSimpleRead(self):
      reader = audio_read.AudioReader(test_filename)
      print reader.done()
      print reader.time()
      print reader.sr
      print reader.duration
      d, sr = reader.read(10)
      print d.shape

  def testReadInPartsNative(self):
      self._testReadInParts(test_filename)

  def testReadInPartsCoercedChans(self):
      self._testReadInParts(test_filename, chans=1)

  def testReadInPartsCoercedSR(self):
      self._testReadInParts(test_filename, sr=8000)

  def testReadWholeNative(self):
      self._testReadWhole(short_test_filename)

  def testReadWholeCoercedChans(self):
      self._testReadWhole(short_test_filename, chans=2)

  def testReadWholeCoercedSR(self):
      self._testReadWhole(short_test_filename, sr=8000)


if __name__ == '__main__':
    unittest.main()
