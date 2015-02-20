"""Find cars passing by in long-duration recordings of Karongwe park."""

from __future__ import print_function
from __future__ import division

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa

from itertools import izip

def vehicle_events_env(d, sr, smooth_time=1.0, n_fft=256):
  """Identify vehicle-passing events in karongwe recordings.
  :args:
    d: np.array
      input audio waveform
    sr: float
      sampling rate of input
  """
  # Calculate spectrogram
  #n_fft = 256
  hop_length = n_fft/2
  D = librosa.stft(d, n_fft=n_fft, hop_length=hop_length)
  frame_rate = sr / float(hop_length)
  # Smooth every row
  #smooth_time = 1.0
  # Make sure the smoothing window length is odd
  smooth_len_frames = 1 + 2*int(smooth_time/2.0*frame_rate)
  # Smooth out the spectrogram rows
  DS = np.array([np.convolve(np.hanning(smooth_len_frames), np.abs(Drow),
                             mode='same')
                 for Drow in D])
  # Further downsample after smoothing
  samples_per_smoo_win = 20
  downsamp_fact = int(smooth_len_frames/samples_per_smoo_win)
  DS = DS[:,::downsamp_fact]
  smoo_frame_rate = frame_rate / downsamp_fact
  # Sum up log-rows over limited frequency range
  min_freq_hz = 1250.0
  max_freq_hz = 5000.0
  min_bin = int(min_freq_hz/sr * n_fft)
  max_bin = int(max_freq_hz/sr * n_fft)
  env = np.sum(np.log(DS[min_bin:max_bin,]),axis=0)
  #plt.plot(env)
  return env, smoo_frame_rate


def vehicle_find_peaks(env, frame_rate, peak_time_scale=1.0,
                       max_threshold=0.03, ignore_time=0.0):
  """Pick peaks in detection envelope.

  :args:
    ignore_time: float
      optional "guard band": events this close to either end are ignored
  :returns:
    times: np.array
      times in seconds of detected events
    energy: np.array
      an "intensity" value associated with each event.
  """
  # Find peaks
  max_win_sec = 0.5 * peak_time_scale
  avg_win_sec = 1.0 * peak_time_scale
  min_sep_sec = 0.3 * peak_time_scale
  max_win_frames = int(round(max_win_sec * frame_rate))
  avg_win_frames = int(round(avg_win_sec * frame_rate))
  min_sep_frames = int(round(min_sep_sec * frame_rate))
  # Threshold for accepting a local max
  delta = np.max(env) * max_threshold
  #print max_win_frames, avg_win_frames, min_sep_frames, delta
  pks = librosa.peak_pick(np.maximum(0,env),
                          pre_max=max_win_frames, post_max=max_win_frames,
                          pre_avg=avg_win_frames, post_avg=avg_win_frames,
                          delta=delta, wait=min_sep_frames)
  ignore_frames = round(ignore_time * frame_rate)
  goodpeaks = np.nonzero((pks > ignore_frames) &
                         (pks < (len(env) - ignore_frames)))
  return pks[goodpeaks].astype(float)/frame_rate, env[pks[goodpeaks]]


def events_for_file(filename, smooth_time=1.0, peak_time_scale=1.0,
                    sensitivity=30.0):
  """ Return all the car events for a file, processing one chunk at a time """
  chunk_dur_sec = 1200.0
  chunk_overlap_sec = 120.0
  current_base_sec = 0.0
  done = False
  sr = 16000
  # Ignore events close to edges, in case we happen to window right on a peak.
  ignore_band_sec = 20.0
  all_times = []
  all_peaks = []
  while not done:
    #print current_base_sec
    d, sr = [d,sr] = librosa.load(filename, sr=sr, offset=current_base_sec,
                                  duration=chunk_dur_sec)
    print("Read %s @ time=%.1f ... %.1f" % (filename, current_base_sec,
                                            current_base_sec + len(d)/sr))
    if len(d) < np.floor(chunk_dur_sec*sr)-1:
      done = True
    env, frame_rate = vehicle_events_env(d, sr, smooth_time=smooth_time)
    this_times, this_peaks = vehicle_find_peaks(env, frame_rate,
                                                ignore_time=ignore_band_sec,
                                                peak_time_scale=peak_time_scale,
                                                max_threshold=1./sensitivity)
    this_times += current_base_sec
    all_times = np.r_[all_times, this_times]
    all_peaks = np.r_[all_peaks, this_peaks]
    current_base_sec += chunk_dur_sec - chunk_overlap_sec
  # Remove duplicate events
  time_thresh = 0.1
  lasttime = min(all_times) - 2*time_thresh
  times = []
  peaks = []
  for time, peak in sorted(zip(all_times, all_peaks)):
    if time > lasttime + time_thresh:
      lasttime = time
      times.append(time)
      peaks.append(peak)
    else:
      # Duplicate time.  Use new peak if larger
      if peak > peaks[-1]:
        peaks[-1] = peak
  return times, peaks


def write_label_file(filename, starts, ends, labels):
  """Write a three-column label file."""
  with open(filename, "w") as f:
    # f.write("# Label file %s written at %s\n" % (filename, time.ctime()))
    for start, end, label in izip(starts, ends, labels):
      label = 'hit'
      f.write("%.3f\t%.3f\t%s\n" % (start, end, label))


def main(argv):
  """Main routine to find car passing events in soundfile."""
  parser = argparse.ArgumentParser(description="Find car passing events.")
  parser.add_argument('input', type=str, help="input WAV file")
  parser.add_argument('output', type=str, help="output TXT file")
  parser.add_argument('--smooth_time', type=float, default=1.0,
                      help="Smooth envelope with kernel of this duration "
                      "in sec.")
  parser.add_argument('--peak_time_scale', type=float, default=1.0,
                      help="Scale the peak picking time windows by this "
                      "factor.")
  parser.add_argument('--sensitivity', type=float, default=30.0,
                      help="Larger sensitivity means smaller peaks count "
                      "as events.")

  args = parser.parse_args()
  input = args.input
  output = args.output

  times, peaks = events_for_file(
      input, smooth_time=args.smooth_time,
      peak_time_scale=args.peak_time_scale,
      sensitivity=args.sensitivity)

  write_label_file(output, times, times, ["%.2f" % peak for peak in peaks])



# Run the main function if called from the command line.
if __name__ == "__main__":
    import sys
    main(sys.argv)