# `blinking_stimuli.py` — Technical Documentation

A Python stimulus presentation tool for flickering image displays, designed for use in steady-state visual evoked potential (SSVEP) research and related visual neuroscience paradigms.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies and Installation](#2-dependencies-and-installation)
3. [Theoretical Background](#3-theoretical-background)
   - 3.1 [Constant-Period (Frame-Count) Approach](#31-constant-period-frame-count-approach)
   - 3.2 [Frequency-Approximation Approach](#32-frequency-approximation-approach)
   - 3.3 [Choosing Between Scheduling Modes](#33-choosing-between-scheduling-modes)
4. [Software Architecture](#4-software-architecture)
5. [API Reference](#5-api-reference)
   - 5.1 [`FlickerMode`](#51-flickermode)
   - 5.2 [`FlickerType`](#52-flickertype)
   - 5.3 [`ColorMode`](#53-colormode)
   - 5.4 [`Stimulus`](#54-stimulus)
   - 5.5 [`StimulusDisplay`](#55-stimulusdisplay)
   - 5.6 [`QuickLayout`](#56-quicklayout)
   - 5.7 [Image Processing Helpers](#57-image-processing-helpers)
6. [Configuration and Usage](#6-configuration-and-usage)
7. [Frequency Constraints by Refresh Rate](#7-frequency-constraints-by-refresh-rate)
8. [Worked Examples](#8-worked-examples)
9. [Startup Diagnostics](#9-startup-diagnostics)
10. [Methodological Considerations](#10-methodological-considerations)
11. [Citation](#11-citation)

---

## 1. Overview

`blinking_stimuli.py` renders one or more image stimuli on a fullscreen black background, each flickering independently at a configurable frequency. It is intended as a lightweight, self-contained stimulus presentation tool for laboratory settings, particularly for EEG paradigms that rely on frequency-tagged visual stimulation such as SSVEP-based brain-computer interfaces (BCIs).

Each stimulus is configured independently along five dimensions:

- **Scheduling mode** — how the on/off timing is computed frame-by-frame (`FlickerMode`)
- **Flicker type** — what is shown during the OFF half-cycle: blank screen or colour negative (`FlickerType`)
- **Colour mode** — whether the image is presented in full colour or greyscale (`ColorMode`)
- **Frequency and phase** — the target frequency in Hz and an optional initial phase offset
- **Pixelation** — an optional mosaic effect that reduces image resolution before flicker generation

Stimuli can be defined individually via the `Stimulus` class, or generated automatically from common spatial arrangements using the `QuickLayout` factory class, which provides built-in grid, circle, and checkerboard layouts.

All flicker scheduling is frame-synchronous; no wall-clock timers are used. This ensures the stimulus sequence is always integer-aligned to the display's vertical refresh cycle.

---

## 2. Dependencies and Installation

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | |
| pygame | ≥ 2.1 (recommended) | `get_current_refresh_rate()` requires ≥ 2.1 |
| numpy | optional | Accelerates greyscale conversion and negative generation; a pure-pygame fallback is used if absent |

Install pygame with pip:

```bash
pip install pygame
```

To enable the faster numpy-accelerated image processing path:

```bash
pip install numpy
```

---

## 3. Theoretical Background

All monitor-based SSVEP stimulators face the same fundamental constraint: because each frame is displayed for exactly `1 / refresh_rate` seconds, only frequencies whose period is an integer multiple of the frame duration can be represented exactly. The two scheduling modes implement different strategies for handling this constraint.

### 3.1 Constant-Period (Frame-Count) Approach

In the conventional approach, each half-period (ON or OFF phase) is assigned a fixed number of frames:

$$n_{half} = \text{round}\!\left(\frac{f_s}{2 f}\right)$$

where $f_s$ is the display refresh rate (Hz) and $f$ is the target stimulus frequency (Hz). The stimulus is visible for $n_{half}$ consecutive frames, then invisible for $n_{half}$ frames, and so on. The true delivered frequency is:

$$f_{actual} = \frac{f_s}{2 \, n_{half}}$$

This is exact only when $f_s / (2f)$ is a whole number. For a 60 Hz display, exact frequencies include 30, 15, 12, 10, 7.5, 6, 5, and 4 Hz. Requesting 11 Hz at 60 Hz yields $n_{half} = 3$, giving $f_{actual} = 10$ Hz — a 9.1% deviation. The program prints a console warning for any deviation exceeding 1%.

This mode produces a perfectly regular, periodic waveform and is appropriate when the target frequency can be chosen to match an achievable value for the given refresh rate.

### 3.2 Frequency-Approximation Approach

The approximation method was proposed by Wang et al. (2010) and comprehensively validated by Nakanishi et al. (2014). At each frame $i$, the on/off state of the stimulus is determined by:

$$s[i] = \begin{cases} 1 & \text{if } \left(\dfrac{i \cdot f}{f_s} + \varphi\right) \bmod 1 < 0.5 \\[6pt] 0 & \text{otherwise} \end{cases}$$

where $f$ is the target frequency, $f_s$ is the refresh rate, and $\varphi \in [0, 1)$ is an optional initial phase offset. This is Equation (1) from Nakanishi et al. (2014), extended here to include phase.

The formula generates a 50% duty-cycle square wave at *any* target frequency below the Nyquist limit ($f < f_s / 2$) by automatically interleaving periods of slightly different lengths. For example, an 11 Hz signal at 60 Hz alternates between 5-frame and 6-frame half-periods such that the long-run average converges to exactly 11 Hz.

Nakanishi et al. (2014) compared SSVEP characteristics from both approaches using EEG across ten subjects and five stimulus frequencies (9–13 Hz). No statistically significant difference was found between the two methods in SSVEP amplitude, signal-to-noise ratio, phase, latency, scalp distribution, or offline classification accuracy for most frequency-refresh rate combinations. The approximation approach was subsequently validated in a simulated online eight-target BCI achieving a mean information transfer rate of 95.0 bits/min.

### 3.3 Choosing Between Scheduling Modes

| Criterion | `APPROXIMATION` | `FRAME_COUNT` |
|---|---|---|
| Supported frequencies | Any $f < f_s / 2$ | Integer multiples of $f_s / 2n$ only |
| Waveform regularity | Quasi-periodic | Strictly periodic |
| Phase coding support | Yes (via `phase` parameter) | No |
| SSVEP signal quality | Comparable (Nakanishi et al., 2014) | Reference standard |
| Recommended for | Flexible designs, BCI, phase coding | Simple designs with known exact frequencies |

---

## 4. Software Architecture

```
blinking_stimuli.py
│
├── FlickerMode          (enum)   APPROXIMATION | FRAME_COUNT
├── FlickerType          (enum)   ON_OFF | ON_NEGATIVE
├── ColorMode            (enum)   COLOUR | GREYSCALE
│
├── _to_greyscale(surf)  Converts a pygame Surface to greyscale (ITU-R BT.601)
├── _to_negative(surf)   Returns colour-inverted copy of a pygame Surface
├── _to_pixelated(surf, block_size)
│                        Returns a mosaic copy: downscale → nearest-neighbour upscale
│
├── Stimulus             (one per visual target)
│   ├── __init__()       Loads image, applies ColorMode + pixelation, pre-computes OFF surface
│   ├── _configure()     Called by StimulusDisplay; computes frame scheduling
│   ├── _on_phase        (property) True during the ON half-cycle
│   ├── update()         Advances frame counter by 1
│   ├── draw()           Blits ON or OFF image to surface
│   └── actual_freq      (property) True delivered frequency
│
├── StimulusDisplay      (application / event loop)
│   ├── __init__()       Stores stimuli list and display settings
│   ├── _print_startup() Renders auto-sizing configuration table to stdout
│   └── run()            Opens window, configures stimuli, runs render loop
│
└── QuickLayout          (layout factory — returns list[Stimulus])
    ├── grid()           Regular rows × cols rectangular grid
    ├── circle()         N stimuli equally spaced around a circle
    └── checkerboard()   rows × cols grid with two interleaved A/B groups
```

Image processing is performed once at construction time in a fixed pipeline: scale to `size` → greyscale (if selected) → pixelate (if selected) → compute negative (if `ON_NEGATIVE`). Applying pixelation before negative generation means both the ON and OFF surfaces are pixelated at the same block size, ensuring the mosaic boundaries remain spatially stable across the entire flicker cycle.

---

## 5. API Reference

### 5.1 `FlickerMode`

Controls the frame-scheduling algorithm.

| Member | Value | Description |
|---|---|---|
| `FlickerMode.APPROXIMATION` | `"Approx"` | Nakanishi et al. (2014) quasi-periodic method; supports any frequency |
| `FlickerMode.FRAME_COUNT` | `"Frame"` | Conventional constant integer half-period; limited to exact frequencies |

---

### 5.2 `FlickerType`

Controls what is rendered during the OFF half-cycle.

| Member | Value | Description |
|---|---|---|
| `FlickerType.ON_OFF` | `"On/Off"` | OFF phase shows nothing (background colour only) |
| `FlickerType.ON_NEGATIVE` | `"On/Negative"` | OFF phase shows the colour-inverted version of the stimulus image |

The `ON_NEGATIVE` mode provides a higher-contrast alternation and may elicit stronger SSVEP responses in some paradigms. The negative image is pre-computed at startup and stored as a separate surface, so there is no runtime cost.

---

### 5.3 `ColorMode`

Controls whether the image is converted to greyscale before display.

| Member | Value | Description |
|---|---|---|
| `ColorMode.COLOUR` | `"Colour"` | Image displayed as loaded (default) |
| `ColorMode.GREYSCALE` | `"Greyscale"` | Image converted to greyscale using ITU-R BT.601 luminance coefficients |

Greyscale conversion uses the standard luminance formula:

$$L = 0.299 R + 0.587 G + 0.114 B$$

If `FlickerType.ON_NEGATIVE` is also selected, the negative is computed *after* greyscale conversion, so the OFF-phase image is the greyscale negative.

---

### 5.4 `Stimulus`

Represents a single blinking image stimulus.

**Constructor**

```python
Stimulus(
    image_path   : str,
    position     : tuple[int, int],
    size         : tuple[int, int] | None = None,
    target_freq  : float = 1.0,
    flicker_mode : FlickerMode = FlickerMode.APPROXIMATION,
    flicker_type : FlickerType = FlickerType.ON_OFF,
    color_mode   : ColorMode   = ColorMode.COLOUR,
    phase        : float = 0.0,
    pixelate     : int | None = None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | Path to the image file (PNG, JPG, BMP, or any format supported by pygame). |
| `position` | `(int, int)` | Pixel coordinates `(x, y)` of the image centre. The origin `(0, 0)` is the top-left corner of the screen. |
| `size` | `(int, int)` or `None` | Target display size `(width, height)` in pixels. `None` preserves original resolution. Scaling uses bilinear interpolation (`smoothscale`). |
| `target_freq` | `float` | Desired flicker frequency in Hz. Must be less than half the display refresh rate. |
| `flicker_mode` | `FlickerMode` | Frame-scheduling algorithm. Defaults to `FlickerMode.APPROXIMATION`. |
| `flicker_type` | `FlickerType` | What to display during the OFF half-cycle. Defaults to `FlickerType.ON_OFF`. |
| `color_mode` | `ColorMode` | Whether to convert the image to greyscale. Defaults to `ColorMode.COLOUR`. |
| `phase` | `float` | Initial phase offset in `[0.0, 1.0)`. `0.0` begins in the ON half-cycle; `0.5` begins in the OFF half-cycle. Only used in `APPROXIMATION` mode. |
| `pixelate` | `int` or `None` | Block size in pixels for a mosaic (pixelation) effect. Each `block_size × block_size` region is averaged to a single colour. `None` or `1` disables pixelation (default: `None`). Applied before flicker generation, so both ON and OFF surfaces are pixelated at the same resolution. |

**Properties**

| Name | Type | Description |
|---|---|---|
| `actual_freq` | `float` | The true frequency delivered to the display. Equals `target_freq` in `APPROXIMATION` mode; quantised in `FRAME_COUNT` mode. Available after `_configure()` is called. |

**Methods**

| Method | Description |
|---|---|
| `update()` | Advance the internal frame counter by one. Call once per display frame, after `draw()`. |
| `draw(surface)` | Blit the ON image (or OFF image for `ON_NEGATIVE`) onto `surface` based on the current frame. |

---

### 5.5 `StimulusDisplay`

Manages the pygame window and the main rendering loop.

**Constructor**

```python
StimulusDisplay(
    stimuli    : list[Stimulus],
    fullscreen : bool  = True,
    fps        : int   = 60,
    bg_color   : tuple = (0, 0, 0),
)
```

| Parameter | Type | Description |
|---|---|---|
| `stimuli` | `list[Stimulus]` | The stimuli to render each frame. |
| `fullscreen` | `bool` | Opens a borderless fullscreen window if `True`; a 1280×720 window if `False`. |
| `fps` | `int` | Target frame rate. Must match the monitor's native refresh rate for accurate timing. |
| `bg_color` | `(int, int, int)` | RGB background colour. Defaults to black `(0, 0, 0)`. |

**Methods**

| Method | Description |
|---|---|
| `run()` | Opens the display (or reuses an existing surface), configures all stimuli, prints the startup table, and enters the render loop. Blocking; does not return. Press **Escape** or close the window to exit. |

---

### 5.6 `QuickLayout`

Factory class for generating `Stimulus` lists in common spatial arrangements. All methods return a `list[Stimulus]` that can be passed directly to `StimulusDisplay`.

Per-stimulus parameters (`image_paths`, `target_freq`, `phase`) accept either a **single scalar value** (applied identically to every stimulus) or a **list** (cycled via modular indexing if shorter than the total number of stimuli).

---

#### `QuickLayout.grid()`

```python
QuickLayout.grid(
    image_paths,
    W: int,
    H: int,
    rows: int,
    cols: int,
    size: tuple         = (150, 150),
    target_freq         = 10.0,
    margin: float       = 0.10,
    flicker_mode        = FlickerMode.APPROXIMATION,
    flicker_type        = FlickerType.ON_OFF,
    color_mode          = ColorMode.COLOUR,
    phase               = 0.0,
    pixelate            = None,
) -> list[Stimulus]
```

Arranges stimuli in a regular `rows × cols` rectangular grid. Positions are computed as the centres of equal-area cells within the usable screen area (screen minus margins). Stimuli are ordered left-to-right, top-to-bottom.

| Parameter | Description |
|---|---|
| `image_paths` | Image path(s). A single string is shared across all positions; a list is cycled. |
| `W`, `H` | Screen width and height in pixels. |
| `rows`, `cols` | Grid dimensions. |
| `size` | Stimulus display size `(width, height)` in pixels. |
| `target_freq` | Flicker frequency in Hz. Scalar or list (cycled). |
| `margin` | Fractional screen margin reserved on each edge (0.0–0.5). E.g. `0.10` leaves a 10% border. |
| `flicker_mode`, `flicker_type`, `color_mode`, `phase` | Stimulus parameters. All accept a scalar or a list (cycled). |
| `pixelate` | Block size for pixelation in pixels. `None` disables. Scalar or list (cycled). |

---

#### `QuickLayout.circle()`

```python
QuickLayout.circle(
    image_paths,
    W: int,
    H: int,
    n: int,
    radius: float       = 0.35,
    center: tuple       = None,
    start_angle: float  = 90.0,
    size: tuple         = (150, 150),
    target_freq         = 10.0,
    flicker_mode        = FlickerMode.APPROXIMATION,
    flicker_type        = FlickerType.ON_OFF,
    color_mode          = ColorMode.COLOUR,
    phase               = 0.0,
    pixelate            = None,
) -> list[Stimulus]
```

Arranges `n` stimuli equally spaced around a circle. Stimuli are ordered clockwise from `start_angle`.

| Parameter | Description |
|---|---|
| `image_paths` | Image path(s). Scalar or list (cycled). |
| `W`, `H` | Screen width and height in pixels. |
| `n` | Number of stimuli. |
| `radius` | Circle radius as a fraction of `min(W, H) / 2`. E.g. `0.35` uses 35% of the half-screen dimension. |
| `center` | `(x, y)` pixel coordinates of the circle centre. Defaults to `(W // 2, H // 2)`. |
| `start_angle` | Angle in degrees where the first stimulus is placed. `90°` = top (12 o'clock). Stimuli proceed clockwise. |
| `target_freq` | Flicker frequency in Hz. Scalar or list (cycled). |
| `flicker_mode`, `flicker_type`, `color_mode`, `phase` | Stimulus parameters. All accept a scalar or a list (cycled). |
| `pixelate` | Block size for pixelation in pixels. `None` disables. Scalar or list (cycled). |

---

#### `QuickLayout.checkerboard()`

```python
QuickLayout.checkerboard(
    image_paths_a,
    image_paths_b,
    W: int,
    H: int,
    rows: int,
    cols: int,
    size: tuple         = (150, 150),
    target_freq_a       = 10.0,
    target_freq_b       = 12.0,
    margin: float       = 0.10,
    flicker_mode        = FlickerMode.APPROXIMATION,
    flicker_type        = FlickerType.ON_OFF,
    color_mode          = ColorMode.COLOUR,
    phase_a             = 0.0,
    phase_b             = 0.0,
    pixelate_a          = None,
    pixelate_b          = None,
) -> list[Stimulus]
```

Arranges stimuli in a `rows × cols` grid where cells alternate between two groups (A and B) in a checkerboard pattern. Group A occupies cells where `(row + col) % 2 == 0`; group B occupies cells where `(row + col) % 2 == 1`:

```
A  B  A  B
B  A  B  A
A  B  A  B
```

This is particularly suited to SSVEP paradigms where two interleaved target sets flicker at different frequencies (or with different images), as the spatial interleaving prevents adaptation to a single spatial location.

| Parameter | Description |
|---|---|
| `image_paths_a`, `image_paths_b` | Image paths for group A and group B. Each accepts a scalar or list (cycled within its group). |
| `W`, `H` | Screen width and height in pixels. |
| `rows`, `cols` | Grid dimensions. |
| `size` | Stimulus display size in pixels. |
| `target_freq_a`, `target_freq_b` | Flicker frequencies for group A and group B. Each accepts a scalar or list cycled within its group. |
| `margin` | Fractional screen margin on each edge (0.0–0.5). |
| `flicker_mode`, `flicker_type`, `color_mode` | Shared stimulus parameters for all cells. |
| `phase_a`, `phase_b` | Phase offsets for group A and group B. Each accepts a scalar or list. |
| `pixelate_a`, `pixelate_b` | Block sizes for pixelation for group A and group B respectively. `None` disables for that group. Each accepts a scalar or list cycled within its group. |

---

### 5.7 Image Processing Helpers

These module-level functions are called internally by `Stimulus.__init__()` but can also be used independently on any `pygame.Surface`.

```python
_to_greyscale(surf: pygame.Surface) -> pygame.Surface
```
Returns a greyscale copy of `surf` using ITU-R BT.601 luminance coefficients, preserving per-pixel alpha. Uses `pygame.surfarray` (numpy) if available; falls back to a pure-pygame pixel loop otherwise.

```python
_to_negative(surf: pygame.Surface) -> pygame.Surface
```
Returns a colour-inverted copy of `surf` — RGB channels replaced by `255 − R`, `255 − G`, `255 − B` — with alpha preserved. Uses `pygame.surfarray` (numpy) if available; falls back to a pure-pygame pixel loop otherwise.

```python
_to_pixelated(surf: pygame.Surface, block_size: int) -> pygame.Surface
```
Returns a pixelated copy of `surf` at the same display dimensions. The image is downscaled to `ceil(w / block_size) × ceil(h / block_size)` using bilinear interpolation, then upscaled back to `w × h` using nearest-neighbour scaling to preserve hard block edges. `block_size ≤ 1` returns an unmodified copy.

---

## 6. Configuration and Usage

**Step 1 — Install dependencies**

```bash
pip install pygame          # required
pip install numpy           # optional, accelerates image processing
```

**Step 2 — Open `blinking_stimuli.py`** and locate the entry-point block at the bottom of the file (`if __name__ == "__main__":`).

**Step 3 — Set display parameters**

```python
TARGET_FPS = 60    # match your monitor's native refresh rate
FULLSCREEN = True  # False for a windowed development mode
```

**Step 4 — Build your stimulus list**

There are two approaches. Use whichever is more convenient.

*Option A — manual stimulus list:*

```python
STIMULI = [
    Stimulus(
        image_path   = "checkerboard.png",
        position     = (W // 2, H // 2),
        size         = (200, 200),
        target_freq  = 10.0,
        flicker_mode = FlickerMode.APPROXIMATION,
        flicker_type = FlickerType.ON_NEGATIVE,
        color_mode   = ColorMode.GREYSCALE,
        phase        = 0.0,
        pixelate     = 8,    # 8×8 px blocks; None to disable
    ),
]
```

*Option B — `QuickLayout` factory (recommended for standard arrangements):*

```python
# 2 × 4 grid at 8–15 Hz, escalating pixelation across columns
STIMULI = QuickLayout.grid(
    "target.png", W, H, rows=2, cols=4,
    target_freq  = [8, 9, 10, 11, 12, 13, 14, 15],
    flicker_type = FlickerType.ON_NEGATIVE,
    pixelate     = [None, 4, 8, 16, None, 4, 8, 16],
)

# 8 stimuli in a circle at 8–15 Hz, uniform pixelation
STIMULI = QuickLayout.circle(
    "target.png", W, H, n=8,
    target_freq = [8, 9, 10, 11, 12, 13, 14, 15],
    radius      = 0.38,
    pixelate    = 8,
)

# 3 × 4 checkerboard — pixelated faces at 10 Hz, sharp houses at 12 Hz
STIMULI = QuickLayout.checkerboard(
    "face.png", "house.png", W, H, rows=3, cols=4,
    target_freq_a = 10.0,
    target_freq_b = 12.0,
    pixelate_a    = 10,
    pixelate_b    = None,
)
```

> **Important.** `pygame.display.set_mode()` must be called *before* the `Stimulus` list is built whenever stimulus positions are computed from `pygame.display.Info()`. The provided entry-point template handles this automatically.

**Step 5 — Run**

```bash
python blinking_stimuli.py
```

Press **Escape** or close the window to exit. A configuration table is printed to the console at startup (see Section 9).

---

## 7. Frequency Constraints by Refresh Rate

The table below lists exact frequencies available under `FlickerMode.FRAME_COUNT` for common refresh rates. All other frequencies require `FlickerMode.APPROXIMATION`.

| Refresh rate | Exact frequencies (Hz) |
|---|---|
| 60 Hz | 30, 15, 12, 10, 7.5, 6, 5, 4, 3, 2.5, 2, 1.5, 1 |
| 75 Hz | 37.5, 25, 15, 12.5, 7.5, 6.25, 5, 3.75, 3, 2.5, 1.5, 1 |
| 120 Hz | 60, 30, 24, 20, 15, 12, 10, 8.57, 7.5, 6, 5, 4, 3, 2, 1 |
| 144 Hz | 72, 36, 24, 18, 14.4, 12, 9, 8, 6, 4.5, 4, 3, 2, 1 |

---

## 8. Worked Examples

### Example 1 — Single central stimulus

```python
STIMULI = [
    Stimulus(
        image_path   = "checkerboard.png",
        position     = (W // 2, H // 2),
        size         = (300, 300),
        target_freq  = 10.0,
        flicker_mode = FlickerMode.APPROXIMATION,
        flicker_type = FlickerType.ON_OFF,
        color_mode   = ColorMode.COLOUR,
    ),
]
```

A single 10 Hz colour stimulus at the screen centre, alternating between visible and blank.

---

### Example 2 — `QuickLayout.grid`: 2 × 4 frequency-tagged array

```python
STIMULI = QuickLayout.grid(
    image_paths  = "target.png",
    W=W, H=H,
    rows         = 2,
    cols         = 4,
    size         = (150, 150),
    target_freq  = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    flicker_type = FlickerType.ON_NEGATIVE,
    color_mode   = ColorMode.GREYSCALE,
)
```

Eight greyscale targets in a 2 × 4 grid, each at a different frequency from 8–15 Hz with ON/negative flicker. Replicates the eight-target paradigm of Nakanishi et al. (2014) in two lines of code.

---

### Example 3 — `QuickLayout.circle`: ring of 6 targets

```python
STIMULI = QuickLayout.circle(
    image_paths = ["cat.png", "dog.png", "car.png",
                   "face.png", "house.png", "tree.png"],
    W=W, H=H,
    n           = 6,
    radius      = 0.38,
    start_angle = 90.0,
    size        = (140, 140),
    target_freq = [8.0, 9.5, 11.0, 12.5, 14.0, 15.5],
    flicker_type= FlickerType.ON_NEGATIVE,
)
```

Six distinct images arranged clockwise starting from the top of the screen, each at a different frequency. Suitable for object-category SSVEP designs.

---

### Example 4 — `QuickLayout.checkerboard`: two interleaved categories

```python
STIMULI = QuickLayout.checkerboard(
    image_paths_a = "face.png",
    image_paths_b = "house.png",
    W=W, H=H,
    rows          = 3,
    cols          = 4,
    size          = (130, 130),
    target_freq_a = 10.0,
    target_freq_b = 12.0,
    flicker_type  = FlickerType.ON_NEGATIVE,
    color_mode    = ColorMode.GREYSCALE,
)
```

A 3 × 4 checkerboard with face images at 10 Hz interleaved with house images at 12 Hz, enabling two-category SSVEP classification. The spatial interleaving ensures each frequency occupies a distributed and balanced set of screen locations.

---

### Example 5 — Phase-coded array (manual)

```python
STIMULI = [
    Stimulus("img_A.png", (W//4,   H//3),   size=(150,150), target_freq=10.0, phase=0.00,
             flicker_type=FlickerType.ON_NEGATIVE, color_mode=ColorMode.GREYSCALE),
    Stimulus("img_B.png", (3*W//4, H//3),   size=(150,150), target_freq=10.0, phase=0.25,
             flicker_type=FlickerType.ON_NEGATIVE, color_mode=ColorMode.GREYSCALE),
    Stimulus("img_C.png", (W//4,   2*H//3), size=(150,150), target_freq=10.0, phase=0.50,
             flicker_type=FlickerType.ON_NEGATIVE, color_mode=ColorMode.GREYSCALE),
    Stimulus("img_D.png", (3*W//4, 2*H//3), size=(150,150), target_freq=10.0, phase=0.75,
             flicker_type=FlickerType.ON_NEGATIVE, color_mode=ColorMode.GREYSCALE),
]
```

Four greyscale stimuli all at 10 Hz with 90° phase offsets (0, 0.25, 0.5, 0.75 cycles), enabling phase-based target coding as described in Jia et al. (2011). Requires `FlickerMode.APPROXIMATION` (the default).

---

## 9. Startup Diagnostics

On each run, `StimulusDisplay.run()` prints a formatted configuration table to standard output before entering the render loop. This provides a permanent record of the stimulus parameters used for a given session and should be logged alongside EEG recordings.

Example output:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          Blinking Stimuli — Configuration                            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  Display refresh rate     60.0 Hz                                                    │
│  Background colour (RGB)  (0, 0, 0)                                                  │
│  Fullscreen               True                                                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  #  Position    Sched. Mode  Target (Hz)  Actual (Hz)  Flicker      Colour     Phase │
│  ────────────────────────────────────────────────────────────────────────────────── │
│  1  (384, 270)  Approx       8.0000       8.0000       On/Off       Colour     0.000 │
│  2  (768, 270)  Approx       9.0000       9.0000       On/Off       Greyscale  0.000 │
│  3  (384, 810)  Frame        12.0000      12.0000      On/Negative  Colour     0.000 │
│  4  (768, 810)  Frame        13.0000      12.0000      On/Negative  Greyscale  0.000 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                       Warnings                                       │
│  ────────────────────────────────────────────────────────────────────────────────── │
│  Stimulus 4:  FRAME_COUNT: requested 13.000 Hz → actual 12.000 Hz (7.7% deviation). │
│               Consider FlickerMode.APPROXIMATION.                                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

A warnings section is only shown when one or more `FRAME_COUNT` stimuli deviate more than 1% from their requested frequency. If no warnings exist, the warnings section is omitted.

---

## 10. Methodological Considerations

**Monitor synchronisation.** Accurate flicker timing depends on the pygame render loop advancing exactly one frame per display refresh. Setting `TARGET_FPS` to match the monitor's native refresh rate generally achieves this, but it is not a substitute for hardware-level verification. For studies with strict timing requirements, an external photodiode placed at a corner of the screen and recorded on a parallel EEG channel is strongly recommended.

**Duty cycle.** Both scheduling modes produce a 50% duty cycle (equal ON and OFF durations). This is the standard duty cycle used in SSVEP research and the value used by Nakanishi et al. (2014).

**ON_NEGATIVE flicker.** Alternating between a stimulus and its colour negative doubles the luminance contrast change per half-cycle compared to ON_OFF flickering, which alternates between the stimulus and a static black background. This can produce stronger and more reliable SSVEP responses, particularly for stimuli with moderate mean luminance.

**Greyscale conversion.** Conversion uses ITU-R BT.601 luminance coefficients (R: 0.299, G: 0.587, B: 0.114), which are standard for standard-definition video content. These coefficients are appropriate for most laboratory monitor gamuts. The alpha channel (transparency) of the source image is preserved through both greyscale conversion and negative generation.

**Image processing performance.** Both `_to_greyscale()` and `_to_negative()` are executed once at stimulus construction, not per frame. With numpy installed, both complete in well under one millisecond for typical stimulus sizes. Without numpy, a pure-pygame pixel loop is used; for a 200×200 image this takes approximately 0.2–0.8 seconds depending on hardware, which is acceptable at startup.

**Phase parameter.** The `phase` parameter shifts the start of the ON half-cycle by a fraction of one full period, defined in normalised units (0.0–1.0, where 1.0 = one full cycle). A phase of `0.25` corresponds to a 90° shift. This parameter is only active in `APPROXIMATION` mode and is silently ignored in `FRAME_COUNT` mode.

**Pixelation.** The `pixelate` parameter applies a mosaic effect by downscaling the image to `ceil(w / block_size) × ceil(h / block_size)` and then upscaling back with nearest-neighbour interpolation. This intentionally degrades spatial resolution, reducing high-frequency spatial content in the stimulus. Because pixelation is applied before negative generation, both the ON and OFF phase surfaces share identical block boundaries, so the mosaic grid is perceptually stable across the entire flicker cycle — only the colour content alternates, not the spatial structure. Larger block sizes reduce spatial information more aggressively. A block size of 1 or `None` disables the effect entirely.

**Number of stimuli.** There is no hard upper limit on the number of simultaneous stimuli. Performance may degrade with very large images or many stimuli on constrained hardware; all stimuli are rendered synchronously within a single frame.

---

## 11. Citation

If this software is used in published research, please cite the methodological paper on which the approximation scheduling mode is based:

> Nakanishi, M., Wang, Y., Wang, Y.-T., Mitsukura, Y., & Jung, T.-P. (2014). Generating visual flickers for eliciting robust steady-state visual evoked potentials at flexible frequencies using monitor refresh rate. *PLoS ONE*, *9*(6), e99235. https://doi.org/10.1371/journal.pone.0099235

The original approximation method was proposed in:

> Wang, Y., Wang, Y.-T., & Jung, T.-P. (2010). Visual stimulus design for high-rate SSVEP. *Electronics Letters*, *46*(15), 1057–1058.

For phase-coded paradigms, the following may also be relevant:

> Jia, C., Gao, X., Hong, B., & Gao, S. (2011). Frequency and phase mixed coding in SSVEP-based brain-computer interface. *IEEE Transactions on Biomedical Engineering*, *58*(1), 200–206. https://doi.org/10.1109/TBME.2010.2068571

---

*Documentation version: 4.0. Corresponds to `blinking_stimuli.py` with `FlickerType`, `ColorMode`, auto-sizing startup diagnostics, `QuickLayout` factory class, and `pixelate` parameter.*