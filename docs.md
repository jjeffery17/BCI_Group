# `blinking_stimuli.py` — Technical Documentation

A Python stimulus presentation tool for flickering image displays, designed for use in steady-state visual evoked potential (SSVEP) research and related visual neuroscience paradigms.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies and Installation](#2-dependencies-and-installation)
3. [Theoretical Background](#3-theoretical-background)
   - 3.1 [Constant-Period (Frame-Count) Approach](#31-constant-period-frame-count-approach)
   - 3.2 [Frequency-Approximation Approach](#32-frequency-approximation-approach)
   - 3.3 [Choosing Between Modes](#33-choosing-between-modes)
4. [Software Architecture](#4-software-architecture)
5. [API Reference](#5-api-reference)
   - 5.1 [`FlickerMode`](#51-flickermode)
   - 5.2 [`Stimulus`](#52-stimulus)
   - 5.3 [`StimulusDisplay`](#53-stimulusdisplay)
6. [Configuration and Usage](#6-configuration-and-usage)
7. [Frequency Constraints by Refresh Rate](#7-frequency-constraints-by-refresh-rate)
8. [Worked Examples](#8-worked-examples)
9. [Startup Diagnostics](#9-startup-diagnostics)
10. [Methodological Considerations](#10-methodological-considerations)
11. [Citation](#11-citation)

---

## 1. Overview

`blinking_stimuli.py` renders one or more image stimuli on a fullscreen black background, each flickering independently at a configurable frequency. It is intended as a lightweight, self-contained stimulus presentation tool for laboratory settings, particularly for EEG paradigms that rely on frequency-tagged visual stimulation such as SSVEP-based brain-computer interfaces (BCIs).

The program implements two distinct scheduling algorithms for controlling the on/off cycle of each stimulus:

- **`FlickerMode.FRAME_COUNT`** — the conventional constant-period method, in which each half-period is a fixed integer number of display frames.
- **`FlickerMode.APPROXIMATION`** — the frequency-approximation method of Nakanishi et al. (2014), which produces a 50% duty-cycle square wave at any target frequency by interleaving slightly variable period lengths on a frame-by-frame basis.

Both modes operate entirely in synchrony with the display's vertical refresh cycle, with no reliance on wall-clock timers or `time.sleep()`. This ensures that the flicker sequence is always an integer-aligned sequence of frames, which is the correct basis for monitor-driven SSVEP stimulation.

---

## 2. Dependencies and Installation

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pygame | ≥ 2.1 (recommended) |

Install the sole external dependency with pip:

```bash
pip install pygame
```

No additional packages are required. The program uses only the Python standard library (`sys`, `enum`) alongside pygame.

> **Note on refresh rate detection.** `pygame.display.get_current_refresh_rate()` was introduced in pygame 2.1. With earlier versions the program falls back to the `fps` value passed to `StimulusDisplay`, which should be set to match the monitor's native refresh rate manually.

---

## 3. Theoretical Background

All monitor-based SSVEP stimulators face the same fundamental constraint: because each frame is displayed for exactly `1 / refresh_rate` seconds, only frequencies whose period is an integer multiple of the frame duration can be represented exactly. The two modes implemented here address this constraint in different ways.

### 3.1 Constant-Period (Frame-Count) Approach

In the conventional approach, each half-period (ON or OFF phase) is assigned a fixed number of frames:

$$n_{half} = \text{round}\!\left(\frac{f_s}{2 f}\right)$$

where $f_s$ is the display refresh rate (Hz) and $f$ is the target stimulus frequency (Hz). The stimulus is visible for $n_{half}$ consecutive frames, then invisible for $n_{half}$ frames, and so on. The true delivered frequency is therefore:

$$f_{actual} = \frac{f_s}{2 \, n_{half}}$$

This is exact only when $f_s / (2f)$ is a whole number. For a 60 Hz display, exact frequencies include 30, 15, 12, 10, 7.5, 6, 5, 4, 3, 2, and 1 Hz. Attempting to present, for example, 11 Hz at 60 Hz yields $n_{half} = 3$ frames, giving an actual frequency of $60 / 6 = 10$ Hz — a 9.1% deviation. The program prints a console warning for any deviation exceeding 1%.

This mode produces a perfectly regular, periodic waveform and is appropriate when the target frequency can be chosen to match an achievable value for the given refresh rate.

### 3.2 Frequency-Approximation Approach

The approximation method was proposed by Wang et al. (2010) and comprehensively validated by Nakanishi et al. (2014). At each frame $i$, the on/off state of the stimulus is determined by:

$$s[i] = \begin{cases} 1 & \text{if } \left(\dfrac{i \cdot f}{f_s} + \varphi\right) \bmod 1 < 0.5 \\[6pt] 0 & \text{otherwise} \end{cases}$$

where $f$ is the target frequency, $f_s$ is the refresh rate, and $\varphi \in [0, 1)$ is an optional initial phase offset. This is Equation (1) from Nakanishi et al. (2014), extended here to include phase.

The formula generates a 50% duty-cycle square wave at *any* target frequency below the Nyquist limit ($f < f_s / 2$) by automatically interleaving periods of slightly different lengths. For example, an 11 Hz signal at 60 Hz alternates between 5-frame and 6-frame half-periods in a pattern such that the long-run average matches 11 Hz exactly. The waveform is quasi-periodic rather than strictly periodic.

Nakanishi et al. (2014) compared the SSVEP characteristics elicited by both approaches using EEG across ten subjects and five stimulus frequencies (9–13 Hz). They found no statistically significant difference between the two methods in SSVEP amplitude, signal-to-noise ratio (SNR), phase, latency, scalp distribution, or offline classification accuracy for most frequency-refresh rate combinations. The approximation approach was subsequently validated in a simulated online eight-target BCI achieving an average information transfer rate of 95.0 bits/min.

The approximation mode is implemented here as the default because it supports arbitrary frequencies without any modification to target parameters.

### 3.3 Choosing Between Modes

| Criterion | `APPROXIMATION` | `FRAME_COUNT` |
|---|---|---|
| Supported frequencies | Any $f < f_s / 2$ | Integer multiples of $f_s / 2$ only |
| Waveform regularity | Quasi-periodic | Strictly periodic |
| Phase coding support | Yes (via `phase` parameter) | No |
| SSVEP signal quality | Comparable (Nakanishi et al., 2014) | Reference standard |
| Recommended for | Flexible frequency designs, BCI, phase coding | Verification, simple designs with known frequencies |

---

## 4. Software Architecture

The program is organised into three classes and a set of optional demo helper functions.

```
blinking_stimuli.py
│
├── FlickerMode          (enum)
│   ├── APPROXIMATION
│   └── FRAME_COUNT
│
├── Stimulus             (one per visual target)
│   ├── __init__()       – loads image, stores parameters
│   ├── _configure()     – called by StimulusDisplay; computes frame scheduling
│   ├── visible          – (property) ON/OFF state for current frame
│   ├── update()         – advances frame counter by 1
│   ├── draw()           – blits image to surface if visible
│   └── actual_freq      – (property) true delivered frequency
│
└── StimulusDisplay      (application / event loop)
    ├── __init__()       – stores stimuli list and display settings
    └── run()            – opens window, configures stimuli, runs render loop
```

The render loop within `StimulusDisplay.run()` follows the sequence: poll events → clear screen → draw all stimuli → flip display buffer → advance all stimulus frame counters. Updating the frame counter *after* drawing ensures that frame 0 is the first frame actually shown, which is important for phase-accurate onset timing.

---

## 5. API Reference

### 5.1 `FlickerMode`

An `enum.Enum` that selects the scheduling algorithm for a `Stimulus`.

| Member | Value | Description |
|---|---|---|
| `FlickerMode.APPROXIMATION` | `"approximation"` | Nakanishi et al. (2014) quasi-periodic method |
| `FlickerMode.FRAME_COUNT` | `"frame_count"` | Conventional constant integer half-period method |

---

### 5.2 `Stimulus`

Represents a single blinking image stimulus.

**Constructor**

```python
Stimulus(
    image_path: str,
    position: tuple[int, int],
    size: tuple[int, int] | None = None,
    target_freq: float = 1.0,
    mode: FlickerMode = FlickerMode.APPROXIMATION,
    phase: float = 0.0,
)
```

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | Path to the image file (PNG, JPG, BMP, or any format supported by pygame). |
| `position` | `(int, int)` | Pixel coordinates `(x, y)` of the image centre on screen. The origin `(0, 0)` is the top-left corner. |
| `size` | `(int, int)` or `None` | Target display size in pixels `(width, height)`. If `None`, the image is shown at its original resolution. Scaling uses bilinear interpolation (`smoothscale`). |
| `target_freq` | `float` | Desired flicker frequency in Hz. Must be less than half the display refresh rate. |
| `mode` | `FlickerMode` | Scheduling algorithm. Defaults to `FlickerMode.APPROXIMATION`. |
| `phase` | `float` | Initial phase offset in the range `[0.0, 1.0)`. A value of `0.0` (default) means the stimulus begins in its ON half-cycle; `0.5` begins in the OFF half-cycle. Only used in `APPROXIMATION` mode. |

**Properties**

| Name | Type | Description |
|---|---|---|
| `visible` | `bool` | Read-only. `True` when the stimulus should be drawn on the current frame, computed from the current frame index and scheduling mode. |
| `actual_freq` | `float` | Read-only. The true frequency delivered to the display. Equals `target_freq` in `APPROXIMATION` mode; quantised in `FRAME_COUNT` mode. Available after `_configure()` has been called. |

**Methods**

| Method | Description |
|---|---|
| `update()` | Advance the internal frame counter by one. Must be called once per display frame, after `draw()`. |
| `draw(surface)` | Blit the stimulus image onto `surface` if `visible` is `True`. |

---

### 5.3 `StimulusDisplay`

Manages the pygame window and the main rendering loop.

**Constructor**

```python
StimulusDisplay(
    stimuli: list[Stimulus],
    fullscreen: bool = True,
    fps: int = 60,
    bg_color: tuple[int, int, int] = (0, 0, 0),
)
```

| Parameter | Type | Description |
|---|---|---|
| `stimuli` | `list[Stimulus]` | The stimuli to render each frame. |
| `fullscreen` | `bool` | If `True`, opens a borderless fullscreen window. If `False`, opens a 1280×720 windowed display. |
| `fps` | `int` | Target frame rate. Should match the monitor's native refresh rate for accurate timing. Common values: 60, 75, 120, 144. |
| `bg_color` | `(int, int, int)` | RGB background colour. Defaults to black `(0, 0, 0)`. |

**Methods**

| Method | Description |
|---|---|
| `run()` | Opens the display (or reuses an existing surface), configures all stimuli, and enters the render loop. Blocking; does not return. Press **Escape** or close the window to exit. |

---

## 6. Configuration and Usage

**Step 1 — Install pygame**

```bash
pip install pygame
```

**Step 2 — Open `blinking_stimuli.py` and locate the entry point block** (the `if __name__ == "__main__":` section at the bottom of the file).

**Step 3 — Set display parameters**

```python
TARGET_FPS = 60    # match your monitor's native refresh rate
FULLSCREEN = True  # False for a development window
```

**Step 4 — Build your stimulus list**

```python
STIMULI = [
    Stimulus(
        image_path  = "checkerboard.png",
        position    = (W // 2, H // 2),
        size        = (200, 200),
        target_freq = 10.0,
        mode        = FlickerMode.APPROXIMATION,
        phase       = 0.0,
    ),
    Stimulus(
        image_path  = "face.png",
        position    = (W // 4, H // 2),
        size        = (150, 150),
        target_freq = 12.0,
        mode        = FlickerMode.FRAME_COUNT,
    ),
]
```

> **Important.** The `pygame.display.set_mode()` call must occur *before* the `Stimulus` list is built whenever stimulus positions are derived from `pygame.display.Info()`. The provided entry-point template handles this automatically.

**Step 5 — Run the script**

```bash
python blinking_stimuli.py
```

Press **Escape** or close the window to exit. Diagnostic information is printed to the console at startup (see Section 9).

---

## 7. Frequency Constraints by Refresh Rate

The table below lists the exact frequencies available under `FlickerMode.FRAME_COUNT` for common refresh rates. All other frequencies require `FlickerMode.APPROXIMATION`.

| Refresh rate | Exact frequencies (Hz) |
|---|---|
| 60 Hz | 30, 15, 12, 10, 7.5, 6, 5, 4, 3, 2.5, 2, 1.5, 1 |
| 75 Hz | 37.5, 25, 15, 12.5, 9.375 (≈ 9.4), 7.5, 6.25, 5, 3.75, 3, 2.5, 1.5, 1 |
| 120 Hz | 60, 30, 24, 20, 15, 12, 10, 8.57 (≈ 8.6), 7.5, 6, 5, 4, 3, 2, 1 |
| 144 Hz | 72, 36, 24, 18, 14.4, 12, 9, 8, 6, 4.5, 4, 3, 2, 1 |

For frequencies not listed above, use `FlickerMode.APPROXIMATION`. The program will print a warning at startup if a `FRAME_COUNT` stimulus deviates more than 1% from its target frequency.

---

## 8. Worked Examples

### Example 1 — Single central stimulus, approximation mode

```python
STIMULI = [
    Stimulus(
        image_path  = "checkerboard.png",
        position    = (W // 2, H // 2),
        size        = (300, 300),
        target_freq = 10.0,
        mode        = FlickerMode.APPROXIMATION,
    ),
]
```

Produces a 10 Hz flicker at the screen centre. At 60 Hz the frame sequence alternates between 3-frame and 3-frame half-periods — exact, because 60 is divisible by 20.

---

### Example 2 — Four-target SSVEP array with phase coding

```python
STIMULI = [
    Stimulus("img_A.png", position=(W//4,   H//3), size=(150,150), target_freq=10.0, phase=0.00),
    Stimulus("img_B.png", position=(3*W//4, H//3), size=(150,150), target_freq=10.0, phase=0.25),
    Stimulus("img_C.png", position=(W//4,   2*H//3), size=(150,150), target_freq=10.0, phase=0.50),
    Stimulus("img_D.png", position=(3*W//4, 2*H//3), size=(150,150), target_freq=10.0, phase=0.75),
]
```

All four stimuli flicker at 10 Hz but are separated by 90° (0.25 cycle) phase steps, enabling phase-based target coding as described in Jia et al. (2011). Requires `FlickerMode.APPROXIMATION` (the default).

---

### Example 3 — Mixed-mode array (8 targets, 8–15 Hz)

This replicates the eight-target BCI paradigm of Nakanishi et al. (2014):

```python
freqs = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
positions = [ ... ]   # eight (x, y) positions

STIMULI = [
    Stimulus(
        image_path  = f"target_{i+1}.png",
        position    = positions[i],
        size        = (120, 120),
        target_freq = freqs[i],
        mode        = FlickerMode.APPROXIMATION,
    )
    for i in range(8)
]
```

At a 75 Hz refresh rate, all eight frequencies are handled accurately by the approximation method.

---

## 9. Startup Diagnostics

On each run, `StimulusDisplay.run()` prints a summary to standard output before entering the render loop:

```
[StimulusDisplay] Refresh rate: 60.0 Hz
  Stimulus @ (960, 270)  mode=approximation    target=10.0000 Hz  actual=10.0000 Hz  phase=0.000
  Stimulus @ (960, 810)  mode=frame_count      target=13.0000 Hz  actual=12.0000 Hz  phase=0.000
  [WARNING] Stimulus @ (960, 810)  FRAME_COUNT: requested 13.000 Hz → actual 12.000 Hz (7.7 % deviation). Consider FlickerMode.APPROXIMATION for this frequency.
```

This output should be logged alongside EEG recordings to confirm that the stimuli were configured as intended prior to data collection.

---

## 10. Methodological Considerations

**Monitor synchronisation.** Accurate flicker timing depends entirely on the pygame render loop advancing exactly one frame per display refresh. Setting `TARGET_FPS` to match the monitor's native refresh rate and using `pygame.time.Clock.tick()` (as implemented) generally achieves this, but it is not a substitute for hardware-level synchronisation (e.g., photodiode recording). For studies with strict timing requirements, an external photodiode placed at a corner of the screen and recorded on a parallel EEG channel is strongly recommended.

**Duty cycle.** Both modes produce a 50% duty cycle (equal ON and OFF durations). This is the standard duty cycle used in SSVEP research and the value used by Nakanishi et al. (2014). Non-50% duty cycles are not currently supported.

**Phase parameter.** The `phase` parameter shifts the onset of the ON half-cycle by a fraction of one full period. It is defined in normalised units where `1.0` equals one full cycle, so a phase of `0.25` corresponds to a 90° shift. This parameter is only active in `APPROXIMATION` mode.

**Stimulus size and location.** Stimulus position is specified as the pixel coordinates of the image centre. The coordinate origin `(0, 0)` is at the top-left of the screen. Stimulus size should be chosen to subtend an appropriate visual angle for the viewing distance and desired cortical response.

**Number of stimuli.** There is no hard limit on the number of simultaneous stimuli. However, rendering performance may degrade with very large images or a very high number of stimuli on slower hardware. All stimuli are rendered synchronously within a single frame.

---

## 11. Citation

If this software is used in published research, please cite the methodological paper on which the approximation mode is based:

> Nakanishi, M., Wang, Y., Wang, Y.-T., Mitsukura, Y., & Jung, T.-P. (2014). Generating visual flickers for eliciting robust steady-state visual evoked potentials at flexible frequencies using monitor refresh rate. *PLoS ONE*, *9*(6), e99235. https://doi.org/10.1371/journal.pone.0099235

The original approximation method was proposed in:

> Wang, Y., Wang, Y.-T., & Jung, T.-P. (2010). Visual stimulus design for high-rate SSVEP. *Electronics Letters*, *46*(15), 1057–1058.

---

*Documentation version: 1.0. Corresponds to `blinking_stimuli.py` as described in this session.*