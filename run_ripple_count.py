import argparse
import sys
from ripple_counter import (
    load_data, find_motor_running_segments, analyze_frequency,
    preprocess_signal, count_ripples_in_segments, count_zero_crossings,
    plot_results
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
    parser.add_argument("--threshold-pct", type=float, default=10.0,
                        help="Motor-on threshold as %% of max current (default: 10)")

    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    try:
        # 1. Load Data
        time, current = load_data(args.filepath, sheet_name=sheet,
                                  current_col=args.col, fs=args.fs)

        # 2. Detect motor-running segments (ignore start/stop transients)
        segments = find_motor_running_segments(current, fs=args.fs,
                                               threshold_pct=args.threshold_pct)

        if not segments:
            print("\n  ERROR: No motor-running segments found!")
            print("  Try lowering --threshold-pct (e.g. --threshold-pct 5)")
            sys.exit(1)

        # 3. FFT on the longest running segment (cleanest frequency estimate)
        longest_seg = max(segments, key=lambda s: s[1] - s[0])
        seg_signal = current[longest_seg[0]:longest_seg[1]]
        dominant_freq, fft_freqs, fft_mag = analyze_frequency(seg_signal, fs=args.fs)

        # 4. Bandpass filter the FULL signal
        filtered = preprocess_signal(current, fs=args.fs,
                                     lowcut=args.lowcut, highcut=args.highcut,
                                     dominant_freq=dominant_freq)

        # 5. Count ripples ONLY in motor-running segments
        print(f"\n  Counting ripples in motor-running segments only:")
        pk_count, peaks = count_ripples_in_segments(
            filtered, segments, fs=args.fs,
            dominant_freq=dominant_freq, prominence=args.prominence
        )

        print(f"\n{'='*50}")
        print(f"  RESULTS")
        print(f"{'='*50}")
        print(f"  Dominant Frequency : {dominant_freq:.1f} Hz")
        print(f"  Peak Detection     : {pk_count} ripples")
        print(f"{'='*50}")

        # 6. Plot
        plot_results(time, current, filtered, fft_freqs, fft_mag,
                     peaks=peaks, segments=segments, dominant_freq=dominant_freq)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
