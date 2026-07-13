import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def load_data(filepath, sheet_name=0, current_col='I', time_col=None, fs=4000):
    """
    Load data from an Excel file.
    """
    print(f"Loading data from '{filepath}' (Sheet: {sheet_name})...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)

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


def analyze_frequency(signal, fs=4000):
    """
    Use FFT to find the dominant ripple frequency in the signal.
    Returns (dominant_freq_hz, fft_freqs, fft_magnitude).
    """
    # Remove DC component before FFT
    signal_centered = signal - np.mean(signal)

    N = len(signal_centered)
    yf = np.abs(rfft(signal_centered))
    xf = rfftfreq(N, 1.0 / fs)

    # Ignore DC bin (index 0) and very low frequencies (< 5 Hz)
    min_freq_idx = np.searchsorted(xf, 5.0)
    # Also ignore frequencies above Nyquist/2 to avoid aliasing artifacts
    max_freq_idx = np.searchsorted(xf, fs / 2 * 0.9)

    search_mag = yf[min_freq_idx:max_freq_idx]
    search_freq = xf[min_freq_idx:max_freq_idx]

    if len(search_mag) == 0:
        return 0, xf, yf

    dominant_idx = np.argmax(search_mag) + min_freq_idx
    dominant_freq = xf[dominant_idx]

    print(f"\n  FFT Analysis:")
    print(f"    Dominant ripple frequency: {dominant_freq:.1f} Hz")

    # Also show top 5 peaks in the spectrum
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
    If lowcut/highcut are not provided, auto-tune them based on the dominant frequency.
    """
    if dominant_freq and dominant_freq > 0:
        if lowcut is None:
            # High-pass at half the dominant freq to remove DC drift
            lowcut = max(1.0, dominant_freq * 0.3)
        if highcut is None:
            # Low-pass at 3x the dominant freq to keep harmonics but remove noise
            highcut = min(dominant_freq * 3.0, fs * 0.45)
    else:
        # Fallback defaults
        if lowcut is None:
            lowcut = 10.0
        if highcut is None:
            highcut = 500.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Clamp to valid range
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


def count_ripples(signal, fs=4000, dominant_freq=None, prominence=None):
    """
    Count ripples using peak detection with auto-tuned parameters.
    A ripple = one up and one down = one peak.
    """
    if dominant_freq and dominant_freq > 0:
        # Minimum distance between peaks: ~80% of one period
        min_distance = int(fs / dominant_freq * 0.6)
        min_distance = max(min_distance, 2)
    else:
        min_distance = 4

    if prominence is None:
        # Auto-set prominence as a fraction of the signal's standard deviation
        std = np.std(signal)
        prominence = std * 0.3
        print(f"  Auto-prominence: {prominence:.4f} (signal std={std:.4f})")

    print(f"  Min peak distance: {min_distance} samples ({fs/max(min_distance,1):.0f} Hz max)")

    peaks, properties = find_peaks(signal, prominence=prominence, distance=min_distance)
    return len(peaks), peaks


def count_zero_crossings(signal):
    """
    Count zero crossings. Ripples = crossings / 2.
    """
    zero_crosses = np.where(np.diff(np.sign(signal)))[0]
    ripple_count = len(zero_crosses) // 2
    return ripple_count, zero_crosses


def plot_results(time, raw_signal, filtered_signal, fft_freqs, fft_mag,
                 peaks=None, dominant_freq=None,
                 title="Motor Current Ripple Analysis"):
    """
    3-panel diagnostic plot: Raw signal, FFT spectrum, Filtered + detected peaks.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # --- Panel 1: Raw Signal ---
    ax1.plot(time, raw_signal, color='gray', linewidth=0.5, alpha=0.8)
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

    # --- Panel 3: Filtered Signal + Peaks ---
    ax3.plot(time, filtered_signal, color='blue', linewidth=0.5, label='Filtered (DC removed)')
    if peaks is not None and len(peaks) > 0:
        ax3.plot(time[peaks], filtered_signal[peaks], "v", color='red',
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
