# Ripple Counter 🔌

A Python tool to count current ripples in brushed DC motor data for sensorless position estimation.

## Quick Start

```bash
pip install -r requirements.txt
python run_ripple_count.py
```

(Defaults are pre-configured for the `002_DX1H_SLP_WithPWM.xlsx` file, sheet `12_PWL_Antipinch`.)

---

## The Method: How We Count Ripples (In Detail)

Motor current contains a lot of information, but it is extremely noisy. Our algorithm extracts the actual commutation ripples using a robust, multi-stage Digital Signal Processing (DSP) pipeline that automatically adapts to the specific motor being tested.

Here is the exact step-by-step breakdown of how the software works:

### 1. Data Loading & Cleaning
When pulling raw data from Excel, motor data exports often contain header rows or string metadata. The script uses pandas `skiprows` to jump past the metadata block (rows 1-5), grabs the specified column, and forcefully drops any remaining non-numeric text (`errors='coerce'`) to guarantee a purely numeric array.

### 2. Fast Fourier Transform (FFT) Auto-Tuning
A major issue with generic ripple counters is that they use hardcoded filter frequencies. Since motor speed (and thus ripple frequency) varies by motor, we auto-tune the filter:
* **High-Pass Pre-filtering (20 Hz):** Before running the FFT, we apply a strict 20 Hz high-pass filter. Motor current has massive low-frequency components (like the slow baseline drift and the huge 0 Hz DC offset). If we don't remove these first, the FFT will falsely identify the baseline drift (~5-10 Hz) as the dominant frequency.
* **Spectrum Analysis:** We run an FFT on this cleaned signal, searching only above 20 Hz to find the true **dominant ripple frequency** (the speed at which the motor's commutator is spinning).

### 3. Adaptive Bandpass Filtering
Once we know the dominant ripple frequency, we design a 4th-order Butterworth bandpass filter tailored *specifically* to that frequency.
* **Low-cut:** Set at 30% of the dominant frequency. This cleanly strips away the slow DC envelope and baseline wander.
* **High-cut:** Set at 300% of the dominant frequency. This keeps the primary ripple and its early harmonics (so the ripple shape isn't distorted) but aggressively destroys high-frequency electrical noise (like PWM switching noise).
* **Zero-Phase Filtering (`filtfilt`):** We pass the signal through the filter forward and backward. This ensures that the peaks do not get shifted in time, which is critical for accurate counting.

### 4. Robust Peak Detection (MAD Thresholding)
The motor starts and stops with massive current spikes (transients) that can be orders of magnitude larger than a normal commutation ripple.
* **Median Absolute Deviation (MAD):** If we set a simple threshold, the start/stop spikes would skew the average so much that normal ripples would be ignored. Instead, we calculate the MAD—a highly robust statistical measure of the "noise floor" of the actual ripples that is entirely immune to extreme outliers (the spikes).
* **Prominence-Based Detection:** We run `scipy.signal.find_peaks` requiring each peak to have a prominence (height relative to its neighboring valleys) of at least 2.5× the MAD. This perfectly isolates real ripples while ignoring electrical "fuzz".
* **Minimum Distance Enforcement:** We strictly enforce that two peaks cannot occur closer than 60% of the expected ripple period (calculated from our FFT dominant frequency). This absolutely prevents a single jagged, noisy ripple from being double-counted as two ripples.

### 5. Four-Panel Diagnostic Plot
Because the algorithm operates on the whole signal simultaneously, visual verification is crucial. The script outputs a 4-panel plot:
1. **Raw Signal:** The original, messy current.
2. **FFT Spectrum:** Shows exactly what frequency the auto-tuner locked onto.
3. **Filtered Signal (Full):** The whole signal with the DC offset removed and all detected peaks marked with dots.
4. **Zoomed View:** Automatically zooms in on a 0.5-second window in the exact center of the recording, drawing clear red circles over the peaks so you can physically verify that the algorithm is correctly locking onto individual ripples.

---

## Why Signal Processing and Not Machine Learning?

### The Simple Explanation

Imagine you're sitting next to a road and you want to count how many cars pass by.

**The ML approach** would be: Record 10,000 hours of video of cars driving by. Label every single car in every frame. Train an AI to recognize cars. Then point the AI at the road and let it count.

**The signal processing approach** would be: Put a speed bump on the road. Every time you feel a bump, add 1 to your counter.

Motor current ripples are like those speed bumps — they are a **predictable, physical event** that happens every time the motor's brushes pass over a commutator segment. We don't need to *teach* a computer what a ripple looks like. We just need to:

1. **Remove the background noise** (like filtering out wind so you can hear each bump clearly)
2. **Count the bumps**

That's literally what a bandpass filter + peak detector does.

### When Would You Actually Need ML?

ML becomes useful when:
- The "bumps" start looking different from each other (variable speed, heavy load changes)
- There are "fake bumps" (electrical noise that looks like a real ripple)
- You want to go beyond counting and predict the exact rotor angle

For the task of "count how many ripples are in this recorded waveform," signal processing is the standard, proven, industry-accepted method.

---

## Industry References

This approach is not something we invented — it is the **exact same method** used by major semiconductor companies in production automotive and industrial systems:

### 1. Texas Instruments — TIDA-01421
**"Automotive Brushed-Motor Ripple Counter Reference Design for Sensorless Position Measurement"**
- Uses an INA240 current sense amplifier → active bandpass filter → comparator with hysteresis → pulse counting on an MSP430 microcontroller
- Designed for automotive power windows, memory seats, power trunks
- [TI Reference Design Page](https://www.ti.com/tool/TIDA-01421)
- [Full Design Guide (PDF)](https://www.ti.com/lit/ug/tidud30a/tidud30a.pdf)

### 2. Microchip — AN3049
**"Sensorless Position Control of Brushed DC Motor Using Ripple Counting Technique"**
- Shunt resistor current sensing → amplification → bandpass filtering → comparator → MCU counter
- Includes source code for PIC/AVR microcontrollers
- [Microchip App Note (PDF)](http://ww1.microchip.com/downloads/en/Appnotes/Sensorless-Position-Control-of-Brushed-DC-Motor-Using-Ripple-Counting-Technique-00003049A.pdf)

### 3. Alps Alpine — HSLRAC Series
**Current Ripple Detection ICs**
- Dedicated silicon chip that does the entire signal conditioning pipeline (sense → filter → output pulses) in a single IC
- Used in automotive slide doors, power windows, seat motors
- [Alps Alpine Product Page](https://www.alpsalpine.com/e/products/category/current-ripple/)

### 4. Open Source — jackw01/ripplecounter
**Hardware + firmware implementation**
- Custom PCB with INA181 current sense amplifier + LM324 bandpass filter + comparator
- Based on the TI and Microchip application notes above
- [GitHub Repository](https://github.com/jackw01/ripplecounter)

### What Our Script Does (Software Equivalent)

Our Python script does **exactly the same thing** as all four references above, but entirely in software on recorded data:

| Hardware Step | Our Software Equivalent |
|---|---|
| Current sense amplifier (INA240/INA181) | `pd.read_excel()` — data already captured |
| Active bandpass filter (Op-Amp circuit) | `scipy.signal.butter()` + `filtfilt()` |
| Comparator with hysteresis | `scipy.signal.find_peaks(prominence=...)` (MAD tuned) |
| MCU pulse counter | `len(peaks)` |

## Usage

```bash
# Run with defaults
python run_ripple_count.py

# Run with a different file
python run_ripple_count.py mydata.xlsx --sheet "Sheet1" --col "current"

# Manual filter tuning (if auto-tune isn't perfect)
python run_ripple_count.py --lowcut 50 --highcut 400 --prominence 0.5
```
