import argparse
import sys
from ripple_counter import load_data, preprocess_signal, count_zero_crossings, count_peaks, plot_results

def main():
    parser = argparse.ArgumentParser(description="Motor Current Ripple Counter")
    parser.add_argument("filepath", type=str, help="Path to the Excel file containing motor data")
    parser.add_argument("--sheet", type=str, default="0", help="Sheet name or index (0-indexed). E.g. 'Sheet1' or '0'")
    parser.add_argument("--col", type=str, default="I", help="Column name for motor current (default: 'I')")
    parser.add_argument("--fs", type=int, default=4000, help="Sampling frequency in Hz (default: 4000)")
    parser.add_argument("--lowcut", type=float, default=10.0, help="Lowcut freq for removing DC offset (default: 10Hz)")
    parser.add_argument("--highcut", type=float, default=500.0, help="Highcut freq for removing jagged noise (default: 500Hz)")
    parser.add_argument("--prominence", type=float, default=0.1, help="Prominence threshold for peak detection")
    
    args = parser.parse_args()
    
    # Handle sheet being an int or a string
    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet
        
    try:
        # 1. Load Data
        time, current = load_data(args.filepath, sheet_name=sheet, current_col=args.col, fs=args.fs)
        
        print(f"Data loaded successfully. Length: {len(current)} samples ({len(current)/args.fs:.2f} seconds)")
        
        # 2. Preprocess (Remove DC offset and smooth the messy 'up and down')
        print(f"Applying bandpass filter ({args.lowcut} Hz - {args.highcut} Hz)...")
        filtered_current = preprocess_signal(current, fs=args.fs, lowcut=args.lowcut, highcut=args.highcut)
        
        # 3. Count Ripples (Method 1: Zero-Crossings)
        zc_ripples, zero_crosses = count_zero_crossings(filtered_current)
        
        # 4. Count Ripples (Method 2: Peak-Valley Prominence)
        # Using a distance of at least 4 samples to prevent double-counting high-frequency noise
        pk_ripples, peaks = count_peaks(filtered_current, prominence=args.prominence, distance=4)
        
        print("\n--- RESULTS ---")
        print(f"Method 1 (Zero-Crossings) : {zc_ripples} ripples")
        print(f"Method 2 (Peak Prominence)  : {pk_ripples} ripples")
        print("----------------")
        
        print("\nGenerating plot for visual verification. Close the plot window to exit.")
        # Plot both peaks and zero crossings on the graph
        plot_results(time, current, filtered_current, peaks=peaks, zero_crosses=zero_crosses)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
