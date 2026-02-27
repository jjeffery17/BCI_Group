"""
Blinking Image Stimuli Display — Pygame
=========================================
Displays a fullscreen black background with configurable image stimuli that
blink independently, each at its own frequency.

Two flicker scheduling modes are supported:

  FlickerMode.APPROXIMATION  (default — recommended for most use cases)
  ─────────────────────────────────────────────────────────────────────
  Based on the frequency-approximation method described in:

      Nakanishi et al. (2014). "Generating Visual Flickers for Eliciting
      Robust Steady-State Visual Evoked Potentials at Flexible Frequencies
      Using Monitor Refresh Rate." PLoS ONE 9(6): e99235.
      https://doi.org/10.1371/journal.pone.0099235

  At every frame i the ON/OFF state is:
      visible = (i * target_freq / refresh_rate + phase) % 1.0 < 0.5

  This creates a 50 % duty-cycle square wave at *any* target frequency
  up to half the refresh rate by interleaving slightly different period
  lengths.

  FlickerMode.FRAME_COUNT  (precise — limited to a discrete set of frequencies)
  ──────────────────────────────────────────────────────────────────────────────
  The conventional constant-period approach. Each half-period is a fixed
  integer number of frames:
      frames_per_half = round(refresh_rate / (2 * target_freq))
  The actual delivered frequency is:
      actual_freq = refresh_rate / (2 * frames_per_half)
  A warning is printed at startup if the deviation exceeds 1 %.

Usage
─────
1.  Install: pip install pygame
2.  Edit the STIMULI list near the bottom of this file.
3.  Run: python blinking_stimuli.py
    Press Escape or close the window to quit.
"""

import sys
import enum
import pygame


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class FlickerMode(enum.Enum):
    """Selects the frame-scheduling algorithm for a Stimulus."""
    APPROXIMATION = "Approx"      # Nakanishi et al. (2014) — flexible freq
    FRAME_COUNT   = "Frame"       # Classic constant-period — exact frames


class FlickerType(enum.Enum):
    """Selects what is shown during the OFF half-cycle."""
    ON_OFF      = "On/Off"        # Alternates between image and blank (black)
    ON_NEGATIVE = "On/Negative"   # Alternates between image and its colour negative


class ColorMode(enum.Enum):
    """Selects whether the image is rendered in full colour or greyscale."""
    COLOUR     = "Colour"
    GREYSCALE  = "Greyscale"


# ──────────────────────────────────────────────────────────────────────────────
# Image processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_greyscale(surf: pygame.Surface) -> pygame.Surface:
    """
    Return a greyscale copy of *surf*, preserving per-pixel alpha.

    Luminance is computed using the ITU-R BT.601 coefficients:
        L = 0.299·R + 0.587·G + 0.114·B

    Uses numpy (via pygame.surfarray) when available for speed; falls back to
    a pure-pygame pixel loop otherwise.
    """
    w, h = surf.get_size()
    out  = pygame.Surface((w, h), pygame.SRCALPHA)

    try:
        import numpy as np
        rgb = pygame.surfarray.array3d(surf)          # shape (w, h, 3), uint8
        lum = (0.299 * rgb[:, :, 0] +
               0.587 * rgb[:, :, 1] +
               0.114 * rgb[:, :, 2]).astype(np.uint8)
        grey_rgb        = np.stack([lum, lum, lum], axis=2)
        pygame.surfarray.blit_array(out, grey_rgb)
        # Copy original alpha channel
        alpha = pygame.surfarray.array_alpha(surf)    # shape (w, h), uint8
        pygame.surfarray.pixels_alpha(out)[:] = alpha

    except ImportError:
        for x in range(w):
            for y in range(h):
                r, g, b, a = surf.get_at((x, y))
                lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                out.set_at((x, y), (lum, lum, lum, a))

    return out


def _to_negative(surf: pygame.Surface) -> pygame.Surface:
    """
    Return a colour-negative copy of *surf* (RGB channels inverted,
    alpha channel preserved).

    Uses numpy (via pygame.surfarray) when available for speed; falls back to
    a pure-pygame pixel loop otherwise.
    """
    w, h = surf.get_size()
    out  = pygame.Surface((w, h), pygame.SRCALPHA)

    try:
        import numpy as np
        rgb = pygame.surfarray.array3d(surf)           # shape (w, h, 3), uint8
        inv_rgb = (255 - rgb).astype("uint8")
        pygame.surfarray.blit_array(out, inv_rgb)
        alpha = pygame.surfarray.array_alpha(surf)
        pygame.surfarray.pixels_alpha(out)[:] = alpha

    except ImportError:
        for x in range(w):
            for y in range(h):
                r, g, b, a = surf.get_at((x, y))
                out.set_at((x, y), (255 - r, 255 - g, 255 - b, a))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stimulus
# ──────────────────────────────────────────────────────────────────────────────

class Stimulus:
    """
    A single blinking image stimulus.

    Parameters
    ----------
    image_path   : str
        Path to an image file (PNG, JPG, BMP, …).
    position     : (int, int)
        (x, y) centre of the image on screen, in pixels.
    size         : (int, int) | None
        (width, height) to scale the image to.  None keeps the original size.
    target_freq  : float
        Desired flicker frequency in Hz.
    flicker_mode : FlickerMode
        APPROXIMATION (default) or FRAME_COUNT — controls frame scheduling.
    flicker_type : FlickerType
        ON_OFF (default) — alternates between image and black.
        ON_NEGATIVE      — alternates between image and its colour negative.
    color_mode   : ColorMode
        COLOUR (default) — display image as loaded.
        GREYSCALE        — convert image to greyscale before display.
    phase        : float
        Initial phase offset in [0.0, 1.0).  0.5 starts in the OFF half-cycle.
        Useful for phase-coded SSVEP paradigms.
        Only applied in APPROXIMATION mode; ignored in FRAME_COUNT mode.
    """

    def __init__(
        self,
        image_path: str,
        position: tuple,
        size: tuple = None,
        target_freq: float = 1.0,
        flicker_mode: FlickerMode = FlickerMode.APPROXIMATION,
        flicker_type: FlickerType = FlickerType.ON_OFF,
        color_mode: ColorMode = ColorMode.COLOUR,
        phase: float = 0.0,
    ):
        self.position     = position
        self.target_freq  = target_freq
        self.flicker_mode = flicker_mode
        self.flicker_type = flicker_type
        self.color_mode   = color_mode
        self.phase        = phase % 1.0

        # ── load and process image ───────────────────────────────────────────
        raw = pygame.image.load(image_path).convert_alpha()
        if size is not None:
            raw = pygame.transform.smoothscale(raw, size)

        if color_mode is ColorMode.GREYSCALE:
            raw = _to_greyscale(raw)

        self._image_on  = raw
        self._image_off = _to_negative(raw) if flicker_type is FlickerType.ON_NEGATIVE else None
        self.rect       = self._image_on.get_rect(center=position)

        # ── frame counter (incremented once per display frame) ───────────────
        self._frame: int = 0

        # ── populated by _configure() before the render loop ────────────────
        self._refresh_rate:    float = 60.0
        self._frames_per_half: int   = 1     # FRAME_COUNT mode only

    # ── called once by StimulusDisplay after the pygame window is open ──────

    def _configure(self, refresh_rate: float) -> None:
        """Derive scheduling parameters from the actual display refresh rate."""
        self._refresh_rate = refresh_rate

        if self.flicker_mode is FlickerMode.FRAME_COUNT:
            self._frames_per_half = max(1, round(refresh_rate / (2.0 * self.target_freq)))
            actual    = refresh_rate / (2.0 * self._frames_per_half)
            deviation = abs(actual - self.target_freq) / self.target_freq
            if deviation > 0.01:
                return (
                    f"FRAME_COUNT: requested {self.target_freq:.3f} Hz → "
                    f"actual {actual:.3f} Hz "
                    f"({deviation * 100:.1f}% deviation). "
                    f"Consider FlickerMode.APPROXIMATION."
                )
        return None  # no warning

    # ── per-frame ON/OFF state ───────────────────────────────────────────────

    @property
    def _on_phase(self) -> bool:
        """True during the ON half-cycle of the current frame."""
        if self.flicker_mode is FlickerMode.APPROXIMATION:
            # Nakanishi et al. (2014), Eq. (1) extended with initial phase φ:
            #   s[i] = square( i·f/fs + φ )
            t = (self._frame * self.target_freq / self._refresh_rate + self.phase) % 1.0
            return t < 0.5
        else:
            return (self._frame % (2 * self._frames_per_half)) < self._frames_per_half

    def update(self) -> None:
        """Advance the internal frame counter by one tick."""
        self._frame += 1

    def draw(self, surface: pygame.Surface) -> None:
        """
        Blit the appropriate image onto *surface* for the current frame.

        ON phase  → draws the (possibly greyscale) stimulus image.
        OFF phase → draws nothing (ON_OFF) or the colour negative (ON_NEGATIVE).
        """
        if self._on_phase:
            surface.blit(self._image_on, self.rect)
        elif self._image_off is not None:
            surface.blit(self._image_off, self.rect)

    # ── convenience property ─────────────────────────────────────────────────

    @property
    def actual_freq(self) -> float:
        """
        The true flicker frequency delivered to the display.

        APPROXIMATION: equals target_freq exactly.
        FRAME_COUNT  : quantised to the nearest achievable frequency.
        """
        if self.flicker_mode is FlickerMode.APPROXIMATION:
            return self.target_freq
        return self._refresh_rate / (2.0 * self._frames_per_half)


# ──────────────────────────────────────────────────────────────────────────────
# Display / application
# ──────────────────────────────────────────────────────────────────────────────

class StimulusDisplay:
    """
    Manages the pygame window and drives a list of Stimulus objects.

    Parameters
    ----------
    stimuli    : list[Stimulus]
        The stimuli to render each frame.
    fullscreen : bool
        Run in fullscreen (True) or in a 1280×720 window (False).
    fps        : int
        Target refresh rate in Hz.  Should match the monitor's native rate.
    bg_color   : (int, int, int)
        RGB background colour (default black).
    """

    def __init__(
        self,
        stimuli: list,
        fullscreen: bool = True,
        fps: int = 60,
        bg_color: tuple = (0, 0, 0),
    ):
        self.stimuli    = stimuli
        self.fps        = fps
        self.bg_color   = bg_color
        self.fullscreen = fullscreen

    # ── print helpers ────────────────────────────────────────────────────────

    def _print_startup(self, refresh_rate: float, warnings: list) -> None:
        """Print a formatted, auto-sizing configuration table to stdout."""

        SEP = "  "   # column separator (two spaces)
        PAD = 2      # inner left/right padding inside the box borders

        # ── build table cell data ────────────────────────────────────────────
        headers = [
            "#", "Position", "Sched. Mode",
            "Target (Hz)", "Actual (Hz)", "Flicker", "Colour", "Phase",
        ]
        rows = [
            [
                str(i),
                str(s.position),
                s.flicker_mode.value,
                f"{s.target_freq:.4f}",
                f"{s.actual_freq:.4f}",
                s.flicker_type.value,
                s.color_mode.value,
                f"{s.phase:.3f}",
            ]
            for i, s in enumerate(self.stimuli, 1)
        ]

        # ── compute column widths from actual content ────────────────────────
        col_w = [len(h) for h in headers]
        for row in rows:
            for j, cell in enumerate(row):
                col_w[j] = max(col_w[j], len(cell))

        def fmt_row(cells: list) -> str:
            """Format a list of cells into a fixed-width column string."""
            return SEP.join(cell.ljust(col_w[j]) for j, cell in enumerate(cells))

        # ── determine box inner width ────────────────────────────────────────
        # Must accommodate: table rows, info lines, and the title
        info_lines = [
            ("Display refresh rate",  f"{refresh_rate:.1f} Hz"),
            ("Background colour (RGB)", str(self.bg_color)),
            ("Fullscreen",            str(self.fullscreen)),
        ]
        info_label_w = max(len(label) for label, _ in info_lines)
        info_strs    = [f"{lbl:<{info_label_w}}  {val}" for lbl, val in info_lines]

        table_row_w = len(fmt_row(headers))
        title       = "Blinking Stimuli — Configuration"

        inner_w = max(
            len(title),
            table_row_w,
            max(len(s) for s in info_strs),
        )

        # ── box drawing helpers ──────────────────────────────────────────────
        total_w = inner_w + PAD * 2   # total width between the │ borders

        def hline(left: str, fill: str, right: str) -> str:
            return left + fill * total_w + right

        def bline(content: str, align: str = "<") -> str:
            """A single bordered line, content padded to inner_w."""
            padded = f"{content:{align}{inner_w}}"
            return f"│{' ' * PAD}{padded}{' ' * PAD}│"

        def divider_line() -> str:
            return f"│{' ' * PAD}{'─' * inner_w}{' ' * PAD}│"

        # ── print ────────────────────────────────────────────────────────────
        print()
        print(hline("┌", "─", "┐"))
        print(bline(title, "^"))
        print(hline("├", "─", "┤"))
        for s in info_strs:
            print(bline(s))
        print(hline("├", "─", "┤"))
        print(bline(fmt_row(headers)))
        print(divider_line())
        for row in rows:
            print(bline(fmt_row(row)))

        if warnings:
            print(hline("├", "─", "┤"))
            print(bline("Warnings", "^"))
            print(divider_line())
            prefix_w  = max(len(f"Stimulus {idx}: ") for idx, _ in warnings)
            wrap_w    = inner_w - prefix_w
            for idx, msg in warnings:
                prefix = f"Stimulus {idx}: ".ljust(prefix_w)
                indent = " " * prefix_w
                # word-wrap the message to fit within the box
                words, out_lines, line = msg.split(), [], ""
                for word in words:
                    candidate = (line + " " + word).strip()
                    if len(candidate) <= wrap_w:
                        line = candidate
                    else:
                        if line:
                            out_lines.append(line)
                        line = word
                if line:
                    out_lines.append(line)
                for k, ln in enumerate(out_lines):
                    print(bline((prefix if k == 0 else indent) + ln))

        print(hline("└", "─", "┘"))
        print()

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the display, configure stimuli, and enter the render loop."""
        pygame.init()

        # Reuse an existing surface (opened before stimuli were built) or create
        # a new one.
        screen = pygame.display.get_surface()
        if screen is None:
            flags = (pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
                     if self.fullscreen else 0)
            if self.fullscreen:
                screen = pygame.display.set_mode((0, 0), flags)
            else:
                screen = pygame.display.set_mode((1280, 720), flags)

        pygame.display.set_caption("Blinking Stimuli")
        pygame.mouse.set_visible(False)

        # Detect the display refresh rate; fall back to the requested fps.
        refresh_rate = float(self.fps)
        try:
            detected = pygame.display.get_current_refresh_rate()   # pygame ≥ 2.1
            if detected > 0:
                refresh_rate = float(detected)
        except AttributeError:
            pass

        # Configure each stimulus and collect any warnings
        warnings = []
        for i, s in enumerate(self.stimuli, 1):
            msg = s._configure(refresh_rate)
            if msg:
                warnings.append((i, msg))

        self._print_startup(refresh_rate, warnings)

        clock = pygame.time.Clock()

        while True:
            clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            screen.fill(self.bg_color)
            for stimulus in self.stimuli:
                stimulus.draw(screen)
                stimulus.update()

            pygame.display.flip()


# ──────────────────────────────────────────────────────────────────────────────
# Demo helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_placeholder(color: tuple, label: str, size: int = 200) -> str:
    """Save a colour-filled square PNG to a temp file and return its path."""
    import os, tempfile
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill(color)
    font = pygame.font.SysFont(None, max(20, size // 7))
    text = font.render(label, True, (255, 255, 255))
    surf.blit(text, text.get_rect(center=(size // 2, size // 2)))
    path = os.path.join(tempfile.gettempdir(),
                        f"ssvep_{label.replace(' ', '_')}.png")
    pygame.image.save(surf, path)
    return path


def create_demo_stimuli(W: int, H: int) -> list:
    """
    Eight demo stimuli: 4 across the top row and 4 across the bottom, covering
    8–15 Hz.  Demonstrates all four combinations of FlickerType and ColorMode.

    Top-left pair    → APPROXIMATION + ON_OFF + COLOUR
    Top-right pair   → APPROXIMATION + ON_OFF + GREYSCALE
    Bottom-left pair → FRAME_COUNT   + ON_NEGATIVE + COLOUR
    Bottom-right pair→ FRAME_COUNT   + ON_NEGATIVE + GREYSCALE
    """
    pygame.font.init()

    specs = [
        #  color              label     Hz    sched mode               flicker type            color mode
        ((220,  60,  60), " 8 Hz",   8.0, FlickerMode.APPROXIMATION, FlickerType.ON_OFF,      ColorMode.COLOUR),
        (( 60, 180,  60), " 9 Hz",   9.0, FlickerMode.APPROXIMATION, FlickerType.ON_OFF,      ColorMode.GREYSCALE),
        (( 60, 120, 220), "10 Hz",  10.0, FlickerMode.APPROXIMATION, FlickerType.ON_NEGATIVE, ColorMode.COLOUR),
        ((200, 160,  40), "11 Hz",  11.0, FlickerMode.APPROXIMATION, FlickerType.ON_NEGATIVE, ColorMode.GREYSCALE),
        ((160,  60, 210), "12 Hz",  12.0, FlickerMode.FRAME_COUNT,   FlickerType.ON_OFF,      ColorMode.COLOUR),
        (( 40, 200, 190), "13 Hz",  13.0, FlickerMode.FRAME_COUNT,   FlickerType.ON_OFF,      ColorMode.GREYSCALE),
        ((220, 120,  40), "14 Hz",  14.0, FlickerMode.FRAME_COUNT,   FlickerType.ON_NEGATIVE, ColorMode.COLOUR),
        ((200, 200, 200), "15 Hz",  15.0, FlickerMode.FRAME_COUNT,   FlickerType.ON_NEGATIVE, ColorMode.GREYSCALE),
    ]

    paths = [_make_placeholder(c, lbl) for c, lbl, *_ in specs]

    positions = [
        (W * 1 // 5, H // 4), (W * 2 // 5, H // 4),
        (W * 3 // 5, H // 4), (W * 4 // 5, H // 4),
        (W * 1 // 5, H * 3 // 4), (W * 2 // 5, H * 3 // 4),
        (W * 3 // 5, H * 3 // 4), (W * 4 // 5, H * 3 // 4),
    ]

    return [
        Stimulus(
            image_path   = paths[i],
            position     = positions[i],
            size         = (160, 160),
            target_freq  = specs[i][2],
            flicker_mode = specs[i][3],
            flicker_type = specs[i][4],
            color_mode   = specs[i][5],
        )
        for i in range(8)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─── CONFIGURE YOUR EXPERIMENT HERE ──────────────────────────────────────
    #
    # Build your STIMULI list and pass it to StimulusDisplay, e.g.:
    #
    #   STIMULI = [
    #       Stimulus(
    #           image_path   = "checkerboard.png",
    #           position     = (W // 2, H // 2),
    #           size         = (200, 200),
    #           target_freq  = 10.0,
    #           flicker_mode = FlickerMode.APPROXIMATION,
    #           flicker_type = FlickerType.ON_NEGATIVE,
    #           color_mode   = ColorMode.GREYSCALE,
    #           phase        = 0.0,
    #       ),
    #   ]
    #
    # ─── FREQUENCY GUIDANCE ──────────────────────────────────────────────────
    #
    # APPROXIMATION: any target_freq < refresh_rate / 2 is valid.
    # FRAME_COUNT:   exact only when refresh_rate / (2 * target_freq) is an
    #                integer. A warning is printed if deviation > 1 %.
    #
    # ─────────────────────────────────────────────────────────────────────────

    TARGET_FPS = 60    # set to your monitor's native refresh rate
    FULLSCREEN = True  # False for a 1280×720 windowed mode

    pygame.init()

    # Open the display FIRST so pygame.display.Info() returns the real screen
    # dimensions, which are needed to position stimuli correctly.
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF if FULLSCREEN else 0
    if FULLSCREEN:
        pygame.display.set_mode((0, 0), flags)
    else:
        pygame.display.set_mode((1280, 720), flags)

    info = pygame.display.Info()
    W, H = info.current_w, info.current_h

    STIMULI = create_demo_stimuli(W, H)

    StimulusDisplay(
        stimuli    = STIMULI,
        fullscreen = FULLSCREEN,
        fps        = TARGET_FPS,
        bg_color   = (0, 0, 0),
    ).run()