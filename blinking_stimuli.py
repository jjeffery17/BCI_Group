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
import re
import math
import enum
import subprocess
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
# Refresh-rate detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_refresh_rate_os() -> float | None:
    """
    Query the primary display's vertical refresh rate using OS-native APIs.

    No third-party packages are required.  Returns the detected rate in Hz, or
    ``None`` if the query fails or the result is not plausible (≤ 10 Hz).

    Platform support
    ────────────────
    Windows  ctypes + ``GetDeviceCaps(hdc, VREFRESH)``
             More reliable than ``EnumDisplaySettings`` because it reads the
             value the graphics driver actually uses for the current mode,
             whereas ``EnumDisplaySettings.dmDisplayFrequency`` may return 0
             or 1 on some hardware/driver combinations.

    Linux    ``xrandr`` subprocess (X11 only).
             Parses the line containing ``*`` (active mode), e.g.
             ``1920x1080   144.00*+``.  Returns ``None`` on Wayland sessions
             where ``xrandr`` is unavailable or reports no active output.

    macOS    ``system_profiler SPDisplaysDataType`` subprocess.
             Matches ``Refresh Rate: 60 Hz`` or ``@ 60.00Hz`` patterns.
             Typical latency is 0.5–2 s; acceptable at startup only.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            VREFRESH = 116                              # GDI GetDeviceCaps index
            hdc = ctypes.windll.user32.GetDC(None)     # DC for the primary display
            if hdc:
                rate = ctypes.windll.gdi32.GetDeviceCaps(hdc, VREFRESH)
                ctypes.windll.user32.ReleaseDC(None, hdc)
                if rate > 10:
                    return float(rate)

        elif sys.platform.startswith("linux"):
            out = subprocess.check_output(
                ["xrandr"], stderr=subprocess.DEVNULL, timeout=3
            ).decode("utf-8", errors="replace")
            # Active mode is marked with '*', e.g. "  1920x1080   144.00*+"
            m = re.search(r"(\d+\.\d+)\s*\*", out)
            if m:
                rate = float(m.group(1))
                if rate > 10:
                    return rate

        elif sys.platform == "darwin":
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL, timeout=8,
            ).decode("utf-8", errors="replace")
            for pattern in (
                r"Refresh Rate:\s+(\d+(?:\.\d+)?)\s+Hz",   # Intel/AMD Macs
                r"@\s*(\d+(?:\.\d+)?)\s*[Hh]z",            # Apple Silicon
            ):
                m = re.search(pattern, out)
                if m:
                    rate = float(m.group(1))
                    if rate > 10:
                        return rate

    except Exception:
        pass

    return None


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
        # Preserve the original image path for external diagnostics
        self.image_path   = image_path
        self.target_freq  = target_freq
        self.flicker_mode = flicker_mode
        self.flicker_type = flicker_type
        self.color_mode   = color_mode
        self.phase        = phase % 1.0

        # ── load and process image ───────────────────────────────────────────
        # Load image without forcing a surface pixel-format conversion.
        # `convert_alpha()` raises "cannot convert without pygame.display
        # initialized" when no display is present (we run under PsychoPy's
        # window). Only call `convert_alpha()` when the pygame display is
        # initialised; otherwise keep the loaded surface as-is.
        raw = pygame.image.load(image_path)
        try:
            if pygame.display.get_init():
                raw = raw.convert_alpha()
        except pygame.error:
            # Fall back to the unconverted surface if conversion fails.
            pass
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

    def _print_startup(self, refresh_rate: float, vsync_ok: bool,
                       warnings: list) -> None:
        """Print a formatted, auto-sizing configuration table to stdout."""

        SEP = "  "   # column separator (two spaces)
        PAD = 2      # inner left/right padding inside the box borders

        # ── split warnings into system-level and per-stimulus ────────────────
        sys_warnings     = [(idx, msg) for idx, msg in warnings if idx is None]
        stim_warnings    = [(idx, msg) for idx, msg in warnings if idx is not None]

        # ── build table cell data ────────────────────────────────────────────
        headers = [
            "#", "Position", "Sched. Mode",
            "Target (Hz)", "Actual (Hz)", "Flicker", "Colour", "Pixelate", "Phase",
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
                str(px) if (px := getattr(s, "pixelate", None)) is not None else "—",
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
        vsync_str  = "enabled (display.flip blocks on retrace)" if vsync_ok \
                     else "not available — using clock.tick() fallback"
        info_lines = [
            ("Display refresh rate",    f"{refresh_rate:.1f} Hz"),
            ("Vsync",                   vsync_str),
            ("Background colour (RGB)", str(self.bg_color)),
            ("Fullscreen",              str(self.fullscreen)),
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

        def _wrap_warning(header: str, msg: str) -> list:
            """Word-wrap *msg* so each line fits inside the box."""
            prefix_w = len(header)
            wrap_w   = inner_w - prefix_w
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
            indent = " " * prefix_w
            result = []
            for k, ln in enumerate(out_lines):
                result.append(bline((header if k == 0 else indent) + ln))
            return result

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

        # ── system warnings (idx = None) — shown before stimulus warnings ────
        if sys_warnings:
            print(hline("├", "─", "┤"))
            print(bline("System Warnings", "^"))
            print(divider_line())
            for _, msg in sys_warnings:
                for line in _wrap_warning("", msg):
                    print(line)

        # ── per-stimulus warnings ─────────────────────────────────────────────
        if stim_warnings:
            print(hline("├", "─", "┤"))
            print(bline("Stimulus Warnings", "^"))
            print(divider_line())
            prefix_w = max(len(f"Stimulus {idx}: ") for idx, _ in stim_warnings)
            for idx, msg in stim_warnings:
                header = f"Stimulus {idx}: ".ljust(prefix_w)
                for line in _wrap_warning(header, msg):
                    print(line)

        print(hline("└", "─", "┘"))
        print()

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the display, configure stimuli, and enter the render loop."""
        pygame.init()

        # ── open the display ─────────────────────────────────────────────────
        # Request vsync=1 so that display.flip() blocks until the monitor's
        # next vertical retrace.  When vsync is active the flip itself paces
        # the loop to exactly one refresh interval — clock.tick() must NOT
        # be called (it would add a redundant sleep on top of the retrace wait,
        # causing every-other-flip to be skipped and halving effective fps).
        # If vsync=1 is unavailable (old pygame / SDL / driver), we fall back
        # to software pacing via clock.tick(fps) only.
        flags    = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF \
                   if self.fullscreen else pygame.DOUBLEBUF
        size     = (0, 0) if self.fullscreen else (1280, 720)
        vsync_ok = False

        # Reuse a surface that was opened before stimuli were built (common
        # pattern in the entry-point block).
        screen = pygame.display.get_surface()
        if screen is None:
            try:
                screen   = pygame.display.set_mode(size, flags, vsync=1)
                vsync_ok = True
            except (TypeError, pygame.error):
                # pygame < 2.0 or driver rejects vsync flag
                screen = pygame.display.set_mode(size, flags)

        pygame.display.set_caption("Blinking Stimuli")
        pygame.mouse.set_visible(False)

        # ── refresh-rate detection ────────────────────────────────────────────
        # Three-tier strategy (highest confidence first):
        #
        #   Tier 1 — pygame.display.get_current_refresh_rate()  (pygame ≥ 2.1)
        #             Can silently return 0 when SDL/driver doesn't populate
        #             the field; requires both a "> 0" AND a "> 10" plausibility
        #             guard.  Can also raise pygame.error (not just AttributeError)
        #             on some backends — both exceptions must be caught.
        #
        #   Tier 2 — OS-native query via _detect_refresh_rate_os()
        #             Uses ctypes/GDI on Windows, xrandr subprocess on Linux,
        #             system_profiler on macOS.  No third-party dependencies.
        #
        #   Tier 3 — Fall back to self.fps with a visible warning.
        warnings     = []
        refresh_rate = None

        # — Tier 1 —
        try:
            detected = pygame.display.get_current_refresh_rate()   # pygame ≥ 2.1
            if detected > 10:                # > 0 is insufficient; 0 = "unknown"
                refresh_rate = float(detected)
        except (AttributeError, pygame.error):
            pass

        # — Tier 2 —
        if refresh_rate is None:
            refresh_rate = _detect_refresh_rate_os()

        # — Tier 3 —
        if refresh_rate is None:
            refresh_rate = float(self.fps)
            warnings.append((
                None,
                f"Display refresh rate could not be detected automatically "
                f"(pygame API unavailable or returned 0; OS-native query also "
                f"failed). Using TARGET_FPS = {self.fps} Hz. Verify this "
                f"matches your monitor's native refresh rate.",
            ))

        # Warn if TARGET_FPS doesn't match the hardware rate.  The loop runs at
        # self.fps (via clock.tick), but stimuli were configured with
        # refresh_rate — a mismatch scales all flicker frequencies incorrectly.
        if abs(refresh_rate - self.fps) > 1.0:
            warnings.append((
                None,
                f"TARGET_FPS ({self.fps} Hz) does not match the detected "
                f"display refresh rate ({refresh_rate:.1f} Hz). All flicker "
                f"frequencies will be miscalculated. Set TARGET_FPS = "
                f"{int(round(refresh_rate))} to match your monitor.",
            ))
            # Use self.fps as the effective scheduling rate so it at least
            # matches what clock.tick actually delivers.
            refresh_rate = float(self.fps)

        # ── configure stimuli ────────────────────────────────────────────────
        for i, s in enumerate(self.stimuli, 1):
            msg = s._configure(refresh_rate)
            if msg:
                warnings.append((i, msg))

        self._print_startup(refresh_rate, vsync_ok, warnings)

        # ── render loop ──────────────────────────────────────────────────────
        # When vsync is active, display.flip() already blocks for one refresh
        # interval — clock.tick() must be skipped to avoid double-pacing.
        # When vsync is absent, clock.tick(fps) is the only pacing mechanism.
        clock = pygame.time.Clock()

        while True:
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

            # flip() blocks on vsync when vsync_ok=True, so frame counters are
            # incremented only after the rendered frame is physically on-screen.
            pygame.display.flip()

            for stimulus in self.stimuli:
                stimulus.update()

            if not vsync_ok:
                clock.tick(self.fps)


# ──────────────────────────────────────────────────────────────────────────────
# Quick layout factory
# ──────────────────────────────────────────────────────────────────────────────

class QuickLayout:
    """
    Factory class for generating Stimulus lists in common spatial arrangements.

    All methods return a ``list[Stimulus]`` that can be passed directly to
    ``StimulusDisplay``.

    Per-stimulus parameters (``image_paths``, ``target_freq``, ``phase``) accept
    either a single value (applied to every stimulus) or a list.  If a list is
    shorter than the number of stimuli it is cycled via modular indexing.

    Layout methods
    ──────────────
    grid          Regular rows × cols rectangular grid.
    circle        N stimuli equally spaced around a circle.
    checkerboard  Rows × cols grid with two interleaved groups (A/B) occupying
                  alternate cells in a checkerboard pattern.
    """

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get(val, i: int):
        """Return val[i % len(val)] for lists, or val for scalars."""
        if isinstance(val, (list, tuple)):
            return val[i % len(val)]
        return val

    @staticmethod
    def _make(image_paths, positions, size, target_freqs, phases,
              flicker_mode, flicker_type, color_mode) -> list:
        """Construct a Stimulus for each position with broadcast parameters."""
        stimuli = []
        for i, pos in enumerate(positions):
            stimuli.append(Stimulus(
                image_path   = QuickLayout._get(image_paths,  i),
                position     = pos,
                size         = size,
                target_freq  = QuickLayout._get(target_freqs, i),
                flicker_mode = QuickLayout._get(flicker_mode, i),
                flicker_type = QuickLayout._get(flicker_type, i),
                color_mode   = QuickLayout._get(color_mode,   i),
                phase        = QuickLayout._get(phases,       i),
            ))
        return stimuli

    # ── grid ─────────────────────────────────────────────────────────────────

    @staticmethod
    def grid(
        image_paths,
        W: int,
        H: int,
        rows: int,
        cols: int,
        size: tuple              = (150, 150),
        target_freq              = 10.0,
        margin: float            = 0.10,
        flicker_mode             = FlickerMode.FRAME_COUNT,
        flicker_type             = FlickerType.ON_OFF,
        color_mode               = ColorMode.COLOUR,
        phase                    = 0.0,
    ) -> list:
        """
        Arrange stimuli in a regular *rows* × *cols* rectangular grid.

        Parameters
        ----------
        image_paths : str or list[str]
            Image path(s).  Cycled across all positions if a list.
        W, H        : int
            Screen width and height in pixels.
        rows, cols  : int
            Number of rows and columns.
        size        : (int, int)
            Stimulus display size in pixels.
        target_freq : float or list[float]
            Flicker frequency in Hz.  Cycled if a list.
        margin      : float
            Fractional margin reserved on each edge of the screen (0.0–0.5).
            E.g. 0.10 leaves 10 % of the screen width/height as a border.
        flicker_mode, flicker_type, color_mode, phase
            Passed directly to each ``Stimulus``.  All accept a scalar
            (shared by every stimulus) or a list (cycled).

        Returns
        -------
        list[Stimulus]
            Stimuli ordered left-to-right, top-to-bottom.

        Example
        -------
        >>> stimuli = QuickLayout.grid(
        ...     "target.png", W, H, rows=2, cols=4,
        ...     target_freq=[8, 9, 10, 11, 12, 13, 14, 15],
        ... )
        """
        x0       = W * margin
        y0       = H * margin
        cell_w   = W * (1 - 2 * margin) / cols
        cell_h   = H * (1 - 2 * margin) / rows
        positions = [
            (int(x0 + (c + 0.5) * cell_w),
             int(y0 + (r + 0.5) * cell_h))
            for r in range(rows)
            for c in range(cols)
        ]
        return QuickLayout._make(image_paths, positions, size,
                                 target_freq, phase,
                                 flicker_mode, flicker_type, color_mode)

    # ── circle ────────────────────────────────────────────────────────────────

    @staticmethod
    def circle(
        image_paths,
        W: int,
        H: int,
        n: int,
        radius: float            = 0.35,
        center: tuple            = None,
        start_angle: float       = 90.0,
        size: tuple              = (150, 150),
        target_freq              = 10.0,
        flicker_mode             = FlickerMode.FRAME_COUNT,
        flicker_type             = FlickerType.ON_OFF,
        color_mode               = ColorMode.COLOUR,
        phase                    = 0.0,
    ) -> list:
        """
        Arrange *n* stimuli equally spaced around a circle.

        Parameters
        ----------
        image_paths  : str or list[str]
            Image path(s).  Cycled across all positions if a list.
        W, H         : int
            Screen width and height in pixels.
        n            : int
            Number of stimuli.
        radius       : float
            Circle radius as a fraction of ``min(W, H) / 2``.
            E.g. 0.35 places stimuli at 35 % of the half-screen dimension.
        center       : (int, int) or None
            Pixel coordinates of the circle centre.  Defaults to the screen
            centre ``(W // 2, H // 2)``.
        start_angle  : float
            Angle in degrees at which the first stimulus is placed.
            90° = top (12 o'clock).  Stimuli are placed clockwise.
        target_freq  : float or list[float]
            Flicker frequency in Hz.  Cycled if a list.
        flicker_mode, flicker_type, color_mode, phase
            Passed directly to each ``Stimulus``.  All accept a scalar
            or a list (cycled).

        Returns
        -------
        list[Stimulus]
            Stimuli ordered clockwise from *start_angle*.

        Example
        -------
        >>> stimuli = QuickLayout.circle(
        ...     "target.png", W, H, n=6,
        ...     target_freq=[8, 9, 10, 11, 12, 13],
        ... )
        """
        cx, cy = center if center is not None else (W // 2, H // 2)
        r      = radius * min(W, H) / 2
        positions = []
        for i in range(n):
            # Clockwise from start_angle; subtract because y increases downward.
            angle_rad = math.radians(start_angle - i * 360.0 / n)
            positions.append((
                int(cx + r * math.cos(angle_rad)),
                int(cy - r * math.sin(angle_rad)),
            ))
        return QuickLayout._make(image_paths, positions, size,
                                 target_freq, phase,
                                 flicker_mode, flicker_type, color_mode)

    # ── checkerboard ─────────────────────────────────────────────────────────

    @staticmethod
    def checkerboard(
        image_paths_a,
        image_paths_b,
        W: int,
        H: int,
        rows: int,
        cols: int,
        size: tuple              = (150, 150),
        target_freq_a            = 10.0,
        target_freq_b            = 12.0,
        margin: float            = 0.10,
        flicker_mode             = FlickerMode.FRAME_COUNT,
        flicker_type             = FlickerType.ON_OFF,
        color_mode               = ColorMode.COLOUR,
        phase_a                  = 0.0,
        phase_b                  = 0.0,
    ) -> list:
        """
        Arrange stimuli in a *rows* × *cols* grid where cells alternate between
        two groups (A and B) in a checkerboard pattern:

            A B A B
            B A B A
            A B A B

        Group A occupies cells where ``(row + col) % 2 == 0``;
        group B occupies cells where ``(row + col) % 2 == 1``.

        This is particularly useful for SSVEP paradigms where two interleaved
        sets of targets flicker at different frequencies.

        Parameters
        ----------
        image_paths_a, image_paths_b : str or list[str]
            Image path(s) for group A and group B respectively.
        W, H          : int
            Screen width and height in pixels.
        rows, cols    : int
            Grid dimensions.
        size          : (int, int)
            Stimulus display size in pixels.
        target_freq_a, target_freq_b : float or list[float]
            Flicker frequencies for each group.  Cycled within each group
            if lists are provided.
        margin        : float
            Fractional screen margin on each edge (0.0–0.5).
        flicker_mode, flicker_type, color_mode
            Shared stimulus parameters applied to all stimuli.
        phase_a, phase_b : float or list[float]
            Phase offsets for group A and group B.

        Returns
        -------
        list[Stimulus]
            Stimuli ordered left-to-right, top-to-bottom, with A/B
            assignment determined by the checkerboard pattern.

        Example
        -------
        >>> stimuli = QuickLayout.checkerboard(
        ...     "face.png", "house.png", W, H, rows=3, cols=4,
        ...     target_freq_a=10.0, target_freq_b=12.0,
        ... )
        """
        x0      = W * margin
        y0      = H * margin
        cell_w  = W * (1 - 2 * margin) / cols
        cell_h  = H * (1 - 2 * margin) / rows

        idx_a = idx_b = 0
        stimuli = []

        for r in range(rows):
            for c in range(cols):
                pos   = (int(x0 + (c + 0.5) * cell_w),
                         int(y0 + (r + 0.5) * cell_h))
                group = (r + c) % 2  # 0 → A, 1 → B

                if group == 0:
                    img   = QuickLayout._get(image_paths_a, idx_a)
                    freq  = QuickLayout._get(target_freq_a, idx_a)
                    ph    = QuickLayout._get(phase_a,       idx_a)
                    idx_a += 1
                else:
                    img   = QuickLayout._get(image_paths_b, idx_b)
                    freq  = QuickLayout._get(target_freq_b, idx_b)
                    ph    = QuickLayout._get(phase_b,       idx_b)
                    idx_b += 1

                i = r * cols + c

                stimuli.append(Stimulus(
                    image_path   = img,
                    position     = pos,
                    size         = size,
                    target_freq  = freq,
                    flicker_mode = QuickLayout._get(flicker_mode, i),
                    flicker_type = QuickLayout._get(flicker_type, i),
                    color_mode   = QuickLayout._get(color_mode,   i),
                    phase        = ph,
                ))

        return stimuli


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
    # Option A — manual stimulus list:
    #
    #   STIMULI = [
    #       Stimulus(
    #           image_path   = "target.png",
    #           position     = (W // 2, H // 2),
    #           size         = (150, 150),
    #           target_freq  = 10.0,
    #           flicker_mode = FlickerMode.APPROXIMATION,
    #           flicker_type = FlickerType.ON_NEGATIVE,
    #           color_mode   = ColorMode.GREYSCALE,
    #           phase        = 0.0,
    #       ),
    #   ]
    #
    # Option B — QuickLayout factory (pick one):
    #
    #   # 2 × 4 grid, one frequency per cell
    #   STIMULI = QuickLayout.grid(
    #       "target.png", W, H, rows=2, cols=4,
    #       target_freq=[8, 9, 10, 11, 12, 13, 14, 15],
    #       flicker_type=FlickerType.ON_NEGATIVE,
    #   )
    #
    #   # 8 stimuli equally spaced around a circle
    #   STIMULI = QuickLayout.circle(
    #       "target.png", W, H, n=8,
    #       target_freq=[8, 9, 10, 11, 12, 13, 14, 15],
    #       radius=0.38,
    #   )
    #
    #   # 3 × 4 checkerboard — two interleaved groups at different frequencies
    #   STIMULI = QuickLayout.checkerboard(
    #       "face.png", "house.png", W, H, rows=3, cols=4,
    #       target_freq_a=10.0, target_freq_b=12.0,
    #   )
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

    # ── DEMO: cycle through layout styles ────────────────────────────────────
    # Change DEMO_LAYOUT to "grid", "circle", or "checkerboard" to preview
    # each style using placeholder images.
    DEMO_LAYOUT = "grid"

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

    pygame.font.init()

    # Build placeholder images for the demo
    colours  = [(220,60,60),(60,180,60),(60,120,220),(200,160,40),
                (160,60,210),(40,200,190),(220,120,40),(200,200,200)]
    freqs_8  = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    paths_8  = [_make_placeholder(colours[i], f"{freqs_8[i]:.0f} Hz")
                for i in range(8)]
    paths_a  = [_make_placeholder((220, 80, 80),  "A")]
    paths_b  = [_make_placeholder((80, 120, 220), "B")]

    if DEMO_LAYOUT == "grid":
        # 2 × 4 grid, 8–15 Hz, approximation mode, on/negative flicker
        STIMULI = QuickLayout.grid(
            image_paths  = paths_8,
            W=W, H=H,
            rows=2, cols=4,
            size         = (150, 150),
            target_freq  = freqs_8,
            flicker_type = FlickerType.ON_NEGATIVE,
            color_mode   = ColorMode.COLOUR,
        )

    elif DEMO_LAYOUT == "circle":
        # 8 stimuli in a circle, 8–15 Hz
        STIMULI = QuickLayout.circle(
            image_paths = paths_8,
            W=W, H=H,
            n           = 8,
            radius      = 0.38,
            start_angle = 90.0,
            size        = (150, 150),
            target_freq = freqs_8,
            flicker_type= FlickerType.ON_NEGATIVE,
        )

    elif DEMO_LAYOUT == "checkerboard":
        # 3 × 4 checkerboard — group A at 10 Hz, group B at 12 Hz
        STIMULI = QuickLayout.checkerboard(
            image_paths_a = paths_a,
            image_paths_b = paths_b,
            W=W, H=H,
            rows=3, cols=4,
            size          = (140, 140),
            target_freq_a = 10.0,
            target_freq_b = 12.0,
            flicker_type  = FlickerType.ON_NEGATIVE,
        )

    else:
        STIMULI = create_demo_stimuli(W, H)

    StimulusDisplay(
        stimuli    = STIMULI,
        fullscreen = FULLSCREEN,
        fps        = TARGET_FPS,
        bg_color   = (0, 0, 0),
    ).run()