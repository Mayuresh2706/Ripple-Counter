# Ripple Counter 🔌

A Python tool to count current ripples in brushed DC motor data for sensorless position estimation.

## Quick Start

```bash
pip install -r requirements.txt
python run_ripple_count.py
```

(Defaults are pre-configured for the `002_DX1H_SLP_WithPWM.xlsx` file, sheet `12_PWL_Antipinch`.)

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
| Comparator with hysteresis | `scipy.signal.find_peaks(prominence=...)` |
| MCU pulse counter | `len(peaks)` |

---

## How It Works

1. **Load** your Excel data
2. **FFT** — Automatically finds the dominant ripple frequency in your signal
3. **Bandpass Filter** — Auto-tunes around the detected frequency to remove DC offset and high-frequency noise
4. **Peak Detection** — Counts peaks with prominence-based thresholding (equivalent to hardware hysteresis)
5. **Plot** — 3-panel diagnostic: raw signal, frequency spectrum, filtered signal with marked peaks

## Usage

```bash
# Run with defaults
python run_ripple_count.py

# Run with a different file
python run_ripple_count.py mydata.xlsx --sheet "Sheet1" --col "current"

# Manual filter tuning (if auto-tune isn't perfect)
python run_ripple_count.py --lowcut 50 --highcut 400 --prominence 0.5
```
