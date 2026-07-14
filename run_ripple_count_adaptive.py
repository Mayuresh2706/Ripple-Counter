import argparse
import sys
from ripple_counter_adaptive import (
    load_data, analyze_frequency,
    preprocess_signal, count_ripples_adaptive,
    plot_results
)

def main():
    parser = argparse.ArgumentParser(
        description="Motor Current Ripple Counter (Adaptive Envelope Threshold Method)")
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
    parser.add_argument("--domfreq", type=float, default=None,
                        help="Manual dominant freq (Hz). Auto-tuned if omitted.")
    parser.add_argument("--window-cycles", type=int, default=10,
                        help="Number of ripple cycles in each sliding window (default: 10)")
    parser.add_argument("--threshold-mult", type=float, default=1.5,
                        help="Local RMS multiplier for threshold (default: 1.5). "
                             "Lower = more sensitive, higher = stricter.")

    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    try:
        # 1. Load Data
        time, current = load_data(args.filepath, sheet_name=sheet,
                                  current_col=args.col, fs=args.fs)

        # 2. FFT — Find dominant frequency
        auto_dom_freq, fft_freqs, fft_mag = analyze_frequency(
            current, fs=args.fs, min_ripple_freq=20.0)
        dominant_freq = args.domfreq if args.domfreq is not None else auto_dom_freq

        # 3. Bandpass filter
        filtered = preprocess_signal(current, fs=args.fs,
                                     lowcut=args.lowcut, highcut=args.highcut,
                                     dominant_freq=dominant_freq)

        # 4. Count ripples using adaptive envelope thresholding
        ripple_count, peaks, troughs, extrema, envelope, adaptive_thresh = \
            count_ripples_adaptive(
                filtered, fs=args.fs, dominant_freq=dominant_freq,
                window_cycles=args.window_cycles,
                threshold_multiplier=args.threshold_mult
            )

        print(f"\n{'='*50}")
        print(f"  RESULTS (Adaptive Envelope Method)")
        print(f"{'='*50}")
        print(f"  Dominant Frequency : {dominant_freq:.1f} Hz")
        print(f"  Peaks Detected     : {len(peaks)}")
        print(f"  Troughs Detected   : {len(troughs)}")
        print(f"  Total Ripples      : {ripple_count}")
        print(f"  Window Cycles      : {args.window_cycles}")
        print(f"  Threshold Mult     : {args.threshold_mult}")
        print(f"{'='*50}")

        # 5. Plot
        plot_results(time, current, filtered, fft_freqs, fft_mag,
                     peaks=peaks, troughs=troughs, ripple_count=ripple_count,
                     dominant_freq=dominant_freq,
                     envelope=envelope, adaptive_thresh=adaptive_thresh)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
