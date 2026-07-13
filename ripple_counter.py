import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def load_data(filepath, sheet_name=0, current_col='I', time_col=None, fs=4000):
    """
    Load data from an Excel file.
    """
    print(f"Loading data from '{filepath}' (Sheet: {sheet_name})...")
    # Row 0 = column names (header), rows 1-5 = metadata → skip them
    df = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=[1, 2, 3, 4, 5])

    if current_col not in df.columns:
        print(f"Column '{current_col}' not found. Available columns:")
        for c in df.columns:
            print(f"  - {c}")
        raise ValueError(f"Column '{current_col}' not found in sheet.")

    # Force conversion to numeric, turning any text/errors into NaN
    current = pd.to_numeric(df[current_col], errors='coerce')

    # Drop NaN rows (header text, missing data) instead of filling with 0
    valid_mask = current.notna()
    current = current[valid_mask].values.astype(float)

    if time_col and time_col in df.columns:
        time = pd.to_numeric(df[time_col], errors='coerce')[valid_mask].values.astype(float)
    else:
        time = np.arange(len(current)) / fs

    print(f"  Loaded {len(current)} valid samples ({len(current)/fs:.2f}s at {fs}Hz)")
    print(f"  Current range: {current.min():.3f} to {current.max():.3f}")
    return time, current


def find_motor_running_segments(signal, fs=4000, threshold_pct=10.0, min_duration_s=0.1):
    """
    Detect segments where the motor is actually running (current is non-trivial).
    Ignores the start/stop transients by trimming edges of each segment.
    
    Args:
        signal: Raw current signal.
        threshold_pct: Percentage of max |current| to consider as "motor on".
        min_duration_s: Minimum segment duration in seconds to consider valid.
        
    Returns:
        List of (start_idx, end_idx) tuples for motor-running segments.
    """
    abs_signal = np.abs(signal)
    threshold = np.max(abs_signal) * (threshold_pct / 100.0)
    
    # Boolean mask: True where motor is running
    running = abs_signal > threshold
    
    # Find contiguous segments
    segments = []
    in_segment = False
    start = 0
    
    for i in range(len(running)):
        if running[i] and not in_segment:
            start = i
            in_segment = True
        elif not running[i] and in_segment:
            end = i
            in_segment = False
            # Check minimum duration
            if (end - start) / fs >= min_duration_s:
                segments.append((start, end))
    
    # Handle segment that runs to end of signal
    if in_segment and (len(running) - start) / fs >= min_duration_s:
        segments.append((start, len(running)))
    
    # Trim edges of each segment to avoid start/stop transients
    trim_samples = int(fs * 0.05)  # trim 50ms from each edge
    trimmed = []
    for s, e in segments:
        s_trim = s + trim_samples
        e_trim = e - trim_samples
        if e_trim > s_trim + fs * min_duration_s:  # still long enough after trimming
            trimmed.append((s_trim, e_trim))
    
    print(f"\n  Motor-running detection:")
    print(f"    Threshold: {threshold:.1f} (={threshold_pct}% of max)")
    print(f"    Found {len(trimmed)} running segment(s):")
    for i, (s, e) in enumerate(trimmed):
        dur = (e - s) / fs
        print(f"      Segment {i+1}: {s/fs:.2f}s - {e/fs:.2f}s ({dur:.2f}s)")
    
    return trimmed


def analyze_frequency(signal, fs=4000, min_ripple_freq=20.0):
    """
    Use FFT to find the dominant ripple frequency in the signal.
    High-pass filters first to remove slow baseline drift.
    """
    nyq = 0.5 * fs
    hp_cutoff = min_ripple_freq / nyq
    hp_cutoff = max(hp_cutoff, 0.001)
    hp_cutoff = min(hp_cutoff, 0.999)
    b_hp, a_hp = butter(4, hp_cutoff, btype='high')
    signal_hp = filtfilt(b_hp, a_hp, signal)

    N = len(signal_hp)
    yf = np.abs(rfft(signal_hp))
    xf = rfftfreq(N, 1.0 / fs)

    min_freq_idx = np.searchsorted(xf, min_ripple_freq)
    max_freq_idx = np.searchsorted(xf, fs / 2 * 0.9)

    search_mag = yf[min_freq_idx:max_freq_idx]
    search_freq = xf[min_freq_idx:max_freq_idx]

    if len(search_mag) == 0:
        return 0, xf, yf

    dominant_idx = np.argmax(search_mag) + min_freq_idx
    dominant_freq = xf[dominant_idx]

    print(f"\n  FFT Analysis (searching above {min_ripple_freq} Hz):")
    print(f"    Dominant ripple frequency: {dominant_freq:.1f} Hz")

    spectrum_peaks, _ = find_peaks(yf[min_freq_idx:max_freq_idx], prominence=np.max(search_mag) * 0.1)
    if len(spectrum_peaks) > 0:
        peak_freqs = search_freq[spectrum_peaks]
        peak_mags = search_mag[spectrum_peaks]
        sorted_idx = np.argsort(peak_mags)[::-1][:5]
        print(f"    Top spectral peaks:")
        for i in sorted_idx:
            print(f"      {peak_freqs[i]:.1f} Hz (magnitude: {peak_mags[i]:.1f})")

    return dominant_freq, xf, yf


def preprocess_signal(signal, fs=4000, lowcut=None, highcut=None, dominant_freq=None, order=4):
    """
    Apply a zero-phase bandpass Butterworth filter.
    Auto-tunes based on dominant frequency if not manually set.
    """
    if dominant_freq and dominant_freq > 0:
        if lowcut is None:
            lowcut = max(1.0, dominant_freq * 0.3)
        if highcut is None:
            highcut = min(dominant_freq * 3.0, fs * 0.45)
    else:
        if lowcut is None:
            lowcut = 10.0
        if highcut is None:
            highcut = 500.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)

    if low >= high:
        print(f"  WARNING: lowcut ({lowcut:.1f}) >= highcut ({highcut:.1f}), using defaults")
        low = 10.0 / nyq
        high = 500.0 / nyq

    print(f"  Bandpass filter: {lowcut:.1f} Hz - {highcut:.1f} Hz (order={order})")

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def count_ripples_in_segments(filtered_signal, segments, fs=4000, dominant_freq=None, prominence=None):
    """
    Count ripples ONLY within motor-running segments, ignoring transients.
    """
    if dominant_freq and dominant_freq > 0:
        min_distance = int(fs / dominant_freq * 0.6)
        min_distance = max(min_distance, 2)
    else:
        min_distance = 4

    total_peaks = []
    
    for i, (seg_start, seg_end) in enumerate(segments):
        seg_signal = filtered_signal[seg_start:seg_end]
        
        if prominence is None:
            # Auto-set prominence per segment based on that segment's noise level
            seg_std = np.std(seg_signal)
            seg_prominence = seg_std * 0.3
        else:
            seg_prominence = prominence
        
        peaks, _ = find_peaks(seg_signal, prominence=seg_prominence, distance=min_distance)
        
        # Convert local segment indices to global indices
        global_peaks = peaks + seg_start
        total_peaks.extend(global_peaks.tolist())
        
        print(f"    Segment {i+1}: {len(peaks)} ripples (prominence={seg_prominence:.4f})")
    
    all_peaks = np.array(total_peaks, dtype=int)
    print(f"  Total ripples across all segments: {len(all_peaks)}")
    print(f"  Min peak distance: {min_distance} samples ({fs/max(min_distance,1):.0f} Hz max)")
    
    return len(all_peaks), all_peaks


def count_zero_crossings(signal):
    """Count zero crossings. Ripples = crossings / 2."""
    zero_crosses = np.where(np.diff(np.sign(signal)))[0]
    ripple_count = len(zero_crosses) // 2
    return ripple_count, zero_crosses


def plot_results(time, raw_signal, filtered_signal, fft_freqs, fft_mag,
                 peaks=None, segments=None, dominant_freq=None,
                 title="Motor Current Ripple Analysis"):
    """
    3-panel diagnostic plot with motor-running segments highlighted.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # --- Panel 1: Raw Signal with segments highlighted ---
    ax1.plot(time, raw_signal, color='gray', linewidth=0.5, alpha=0.8)
    if segments:
        for i, (s, e) in enumerate(segments):
            ax1.axvspan(time[s], time[min(e-1, len(time)-1)], alpha=0.15, color='green',
                        label='Motor Running' if i == 0 else None)
    ax1.axhline(np.mean(raw_signal), color='red', linestyle='--', linewidth=1,
                label=f'DC Mean = {np.mean(raw_signal):.2f}')
    ax1.set_title(f"{title} — Raw Signal")
    ax1.set_ylabel("Current (A)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Panel 2: FFT Spectrum ---
    ax2.plot(fft_freqs, fft_mag, color='purple', linewidth=0.8)
    if dominant_freq and dominant_freq > 0:
        ax2.axvline(dominant_freq, color='red', linestyle='--',
                    label=f'Dominant: {dominant_freq:.1f} Hz')
    ax2.set_title("Frequency Spectrum (FFT)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, min(2000, max(fft_freqs)))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- Panel 3: Filtered Signal + Peaks (only in segments) ---
    ax3.plot(time, filtered_signal, color='blue', linewidth=0.5, alpha=0.4, label='Filtered (all)')
    
    # Highlight the segments being analyzed
    if segments:
        for i, (s, e) in enumerate(segments):
            seg_time = time[s:e]
            seg_filtered = filtered_signal[s:e]
            ax3.plot(seg_time, seg_filtered, color='blue', linewidth=0.8,
                     label='Analyzed Region' if i == 0 else None)
            ax3.axvspan(time[s], time[min(e-1, len(time)-1)], alpha=0.08, color='green')
    
    if peaks is not None and len(peaks) > 0:
        ax3.plot(time[peaks], filtered_signal[peaks], "x", color='red',
                 markersize=6, label=f'Detected Peaks (n={len(peaks)})')
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_title(f"Filtered Signal — Detected Ripples: {len(peaks) if peaks is not None else 0}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Current (A)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("ripple_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: ripple_analysis.png")
    plt.show()
