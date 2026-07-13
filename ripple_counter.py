import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def load_data(filepath, sheet_name=0, current_col='I', time_col=None, fs=4000):
    """
    Load data from an Excel file.
    
    Args:
        filepath (str): Path to the Excel file.
        sheet_name (str/int): Name or index of the sheet to load.
        current_col (str): The name of the column containing the motor current.
        time_col (str): The name of the column containing time (if any).
        fs (int): Sampling frequency (default 4000 Hz).
        
    Returns:
        tuple: (time_array, current_array)
    """
    print(f"Loading data from '{filepath}' (Sheet: {sheet_name})...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Force conversion to numeric, turning any text/errors into NaN
    current = pd.to_numeric(df[current_col], errors='coerce')
    
    # Fill any NaNs (like text headers) with 0 or interpolate them
    current = current.fillna(0).values
    
    if time_col and time_col in df.columns:
        time = df[time_col].values
    else:
        # Generate time array based on sampling frequency
        time = np.arange(len(current)) / fs
        
    return time, current

def preprocess_signal(signal, fs=4000, lowcut=10.0, highcut=500.0, order=4):
    """
    Apply a zero-phase bandpass Butterworth filter.
    
    The high-pass part (lowcut) removes the DC offset and slow drift.
    The low-pass part (highcut) removes the high-frequency "jagged/messy" noise.
    
    Args:
        signal (ndarray): Raw current signal.
        fs (int): Sampling frequency.
        lowcut (float): Lower cutoff frequency in Hz (removes DC drift).
        highcut (float): Upper cutoff frequency in Hz (removes jagged noise).
        order (int): Butterworth filter order.
        
    Returns:
        ndarray: Filtered and centered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design the bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply zero-phase filter (doesn't shift the peaks in time)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def count_zero_crossings(signal):
    """
    Method 1: Count zero crossings on the detrended/centered signal.
    A ripple is "one up, one down", which means it crosses zero twice.
    """
    # Find indices where the sign changes
    zero_crosses = np.where(np.diff(np.sign(signal)))[0]
    
    # Number of ripples is half the number of zero crossings
    ripple_count = len(zero_crosses) // 2
    return ripple_count, zero_crosses

def count_peaks(signal, prominence=0.5, distance=10):
    """
    Method 2: Peak detection using prominence.
    This is highly robust to messy signals and moving baselines.
    
    Args:
        signal (ndarray): Filtered signal.
        prominence (float): Minimum height of a peak relative to its neighboring valleys.
        distance (int): Minimum number of samples between peaks.
        
    Returns:
        tuple: (ripple_count, peak_indices)
    """
    peaks, properties = find_peaks(signal, prominence=prominence, distance=distance)
    ripple_count = len(peaks)
    
    return ripple_count, peaks

def plot_results(time, raw_signal, filtered_signal, peaks=None, zero_crosses=None, title="Motor Current Ripple Analysis"):
    """
    Visualize the original signal, the cleaned signal, and the detected ripples.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Raw Signal
    ax1.plot(time, raw_signal, label='Raw Signal (with DC & Noise)', color='gray', alpha=0.8)
    ax1.set_title(title + " - Raw")
    ax1.set_ylabel("Current")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Filtered Signal
    ax2.plot(time, filtered_signal, label='Filtered & Centered Signal', color='blue')
    
    if peaks is not None and len(peaks) > 0:
        ax2.plot(time[peaks], filtered_signal[peaks], "x", color='red', markersize=8, label=f'Detected Peaks (n={len(peaks)})')
        
    if zero_crosses is not None and len(zero_crosses) > 0:
        # Plot zero crossings as green dots on the zero line
        ax2.plot(time[zero_crosses], np.zeros_like(zero_crosses), ".", color='green', markersize=5, label=f'Zero Crossings (n={len(zero_crosses)})')
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')
        
    ax2.set_title("Filtered Signal - Ripple Detection")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Current")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
