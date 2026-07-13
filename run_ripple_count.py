import argparse
import sys
from ripple_counter import (
    load_data, analyze_frequency, preprocess_signal,
    count_ripples, count_zero_crossings, plot_results
)

def main():
    parser = argparse.ArgumentParser(description="Motor Current Ripple Counter")
    parser.add_argument("filepath", type=str, nargs="?",
                        default="002_DX1H_SLP_WithPWM.xlsx",
                        help="Path to the Excel file")
    parser.add_argument("--sheet", type=str, default="12_PWL_Antipinch",
                        help="Sheet name or 0-indexed number")
    parser.add_argument("--col", type=str,
                        default="apmd_Data.motors[0].input.MotorCurrent",
                        help="Column name for motor current")
    parser.add_argument("--fs", type=int, default=4000,
                        help="Sampling frequency in Hz (default: 4000)")
    parser.add_argument("--lowcut", type=float, default=None,
                        help="Manual lowcut freq (Hz). Auto-tuned if omitted.")
    parser.add_argument("--highcut", type=float, default=None,
                        help="Manual highcut freq (Hz). Auto-tuned if omitted.")
    parser.add_argument("--prominence", type=float, default=None,
                        help="Manual prominence threshold. Auto-tuned if omitted.")

    args = parser.parse_args()

    # Handle sheet being an int or a string
    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    try:
        # 1. Load Data
        time, current = load_data(args.filepath, sheet_name=sheet,
                                  current_col=args.col, fs=args.fs)

        # 2. FFT — Find the dominant ripple frequency
        dominant_freq, fft_freqs, fft_mag = analyze_frequency(current, fs=args.fs)

        # 3. Preprocess — Auto-tune filter based on dominant frequency
        filtered = preprocess_signal(current, fs=args.fs,
                                     lowcut=args.lowcut, highcut=args.highcut,
                                     dominant_freq=dominant_freq)

        # 4. Count Ripples — Peak detection (auto-tuned)
        pk_count, peaks = count_ripples(filtered, fs=args.fs,
                                        dominant_freq=dominant_freq,
                                        prominence=args.prominence)

        # 5. Count Ripples — Zero-crossing method
        zc_count, zero_crosses = count_zero_crossings(filtered)

        print(f"\n{'='*40}")
        print(f"  RESULTS")
        print(f"{'='*40}")
        print(f"  Dominant Frequency : {dominant_freq:.1f} Hz")
        print(f"  Peak Detection     : {pk_count} ripples")
        print(f"  Zero-Crossing      : {zc_count} ripples")
        print(f"{'='*40}")

        # 6. Plot
        plot_results(time, current, filtered, fft_freqs, fft_mag,
                     peaks=peaks, dominant_freq=dominant_freq)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
