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
  lengths. The paper demonstrated that SSVEPs elicited this way are
  statistically comparable to those from the constant-period approach,
  and validated it for 8–15 Hz in an eight-target BCI with a 75 Hz display.

  FlickerMode.FRAME_COUNT  (precise — limited to a discrete set of frequencies)
  ──────────────────────────────────────────────────────────────────────────────
  The conventional constant-period approach. Each half-period is a fixed
  integer number of frames:
      frames_per_half = round(refresh_rate / (2 * target_freq))
  The stimulus is ON for that many frames, then OFF for the same count.
  The *actual* flicker frequency will be:
      actual_freq = refresh_rate / (2 * frames_per_half)
  which may differ from the requested frequency unless refresh_rate is
  evenly divisible by (2 * target_freq). A warning is printed at startup
  if the deviation exceeds 1 %.

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
# Flicker mode
# ──────────────────────────────────────────────────────────────────────────────

class FlickerMode(enum.Enum):
    """Selects the frame-scheduling algorithm for a Stimulus."""
    APPROXIMATION = "approximation"   # Nakanishi et al. (2014) — flexible freq
    FRAME_COUNT   = "frame_count"     # Classic constant-period — exact frames


# ──────────────────────────────────────────────────────────────────────────────
# Stimulus
# ──────────────────────────────────────────────────────────────────────────────

class Stimulus:
    """
    A single blinking image stimulus.

    Parameters
    ----------
    image_path  : str
        Path to an image file (PNG, JPG, BMP, …).
    position    : (int, int)
        (x, y) centre of the image on screen, in pixels.
    size        : (int, int) | None
        (width, height) to scale the image to.  None keeps the original size.
    target_freq : float
        Desired flicker frequency in Hz.
    mode        : FlickerMode
        APPROXIMATION (default) or FRAME_COUNT — see module docstring.
    phase       : float
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
        mode: FlickerMode = FlickerMode.APPROXIMATION,
        phase: float = 0.0,
    ):
        self.position    = position
        self.target_freq = target_freq
        self.mode        = mode
        self.phase       = phase % 1.0

        # Load and optionally resize the image
        raw = pygame.image.load(image_path).convert_alpha()
        if size is not None:
            raw = pygame.transform.smoothscale(raw, size)
        self.image = raw
        self.rect  = self.image.get_rect(center=position)

        # Frame counter — incremented once per display frame
        self._frame: int = 0

        # Populated by _configure() before the loop starts
        self._refresh_rate: float = 60.0
        self._frames_per_half: int = 1   # FRAME_COUNT mode only

    # ── called once by StimulusDisplay after the pygame window is open ──────

    def _configure(self, refresh_rate: float) -> None:
        """Derive scheduling parameters from the actual display refresh rate."""
        self._refresh_rate = refresh_rate

        if self.mode is FlickerMode.FRAME_COUNT:
            self._frames_per_half = max(1, round(refresh_rate / (2.0 * self.target_freq)))
            actual    = refresh_rate / (2.0 * self._frames_per_half)
            deviation = abs(actual - self.target_freq) / self.target_freq
            if deviation > 0.01:
                print(
                    f"  [WARNING] Stimulus @ {self.position}  FRAME_COUNT: "
                    f"requested {self.target_freq:.3f} Hz → "
                    f"actual {actual:.3f} Hz "
                    f"({deviation*100:.1f} % deviation). "
                    f"Consider FlickerMode.APPROXIMATION for this frequency."
                )

    # ── per-frame state ──────────────────────────────────────────────────────

    @property
    def visible(self) -> bool:
        """True when the stimulus should be drawn on the current frame."""
        if self.mode is FlickerMode.APPROXIMATION:
            # Nakanishi et al. (2014), Eq. (1) extended with initial phase φ:
            #   s[i] = square( i·f/fs + φ )
            # The 50 % duty-cycle square wave is ON when its fractional argument
            # lies in [0, 0.5).
            t = (self._frame * self.target_freq / self._refresh_rate + self.phase) % 1.0
            return t < 0.5
        else:
            # FRAME_COUNT: constant integer half-period length.
            return (self._frame % (2 * self._frames_per_half)) < self._frames_per_half

    def update(self) -> None:
        """Advance the internal frame counter by one tick."""
        self._frame += 1

    def draw(self, surface: pygame.Surface) -> None:
        """Blit the image onto *surface* when in the ON phase."""
        if self.visible:
            surface.blit(self.image, self.rect)

    # ── convenience ─────────────────────────────────────────────────────────

    @property
    def actual_freq(self) -> float:
        """
        The true flicker frequency delivered to the display.

        APPROXIMATION mode: equals target_freq (floating-point precision).
        FRAME_COUNT mode  : quantised to the nearest achievable frequency.
        """
        if self.mode is FlickerMode.APPROXIMATION:
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
        Target refresh rate in Hz.  Set this to your monitor's native refresh
        rate for accurate flicker timing (e.g. 60, 75, 120, 144).
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

    def run(self) -> None:
        """Enter the main event loop.  Escape or close the window to quit."""
        pygame.init()

        # Reuse an existing surface (opened before stimuli were built) or
        # create a new one.
        screen = pygame.display.get_surface()
        if screen is None:
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF if self.fullscreen else 0
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

        print(f"\n[StimulusDisplay] Refresh rate: {refresh_rate:.1f} Hz")
        for s in self.stimuli:
            s._configure(refresh_rate)
            print(
                f"  Stimulus @ {s.position}  "
                f"mode={s.mode.value:13s}  "
                f"target={s.target_freq:.4f} Hz  "
                f"actual={s.actual_freq:.4f} Hz  "
                f"phase={s.phase:.3f}"
            )
        print()

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
    path = os.path.join(tempfile.gettempdir(), f"ssvep_{label.replace(' ', '_')}.png")
    pygame.image.save(surf, path)
    return path


def create_demo_stimuli(W: int, H: int) -> list:
    """
    Eight demo stimuli mirroring the eight-target BCI in Nakanishi et al.
    (2014): frequencies 8–15 Hz, four across the top and four across the bottom.

    The top row uses FlickerMode.APPROXIMATION (handles all frequencies).
    The bottom row uses FlickerMode.FRAME_COUNT (exact integer periods).
    At 60 Hz the FRAME_COUNT stimuli deviate slightly from the target and
    startup warnings are printed so you can see the difference.
    """
    pygame.font.init()

    # (color, short label, target Hz, mode, initial phase)
    specs = [
        # ── top row: APPROXIMATION mode ──────────────────────────────────────
        ((220,  60,  60), " 8 Hz\nApprox",  8.0, FlickerMode.APPROXIMATION, 0.00),
        (( 60, 180,  60), " 9 Hz\nApprox",  9.0, FlickerMode.APPROXIMATION, 0.00),
        (( 60, 120, 220), "10 Hz\nApprox", 10.0, FlickerMode.APPROXIMATION, 0.00),
        ((200, 160,  40), "11 Hz\nApprox", 11.0, FlickerMode.APPROXIMATION, 0.00),
        # ── bottom row: FRAME_COUNT mode ─────────────────────────────────────
        ((160,  60, 210), "12 Hz\nFrame",  12.0, FlickerMode.FRAME_COUNT,   0.00),
        (( 40, 200, 190), "13 Hz\nFrame",  13.0, FlickerMode.FRAME_COUNT,   0.00),
        ((220, 120,  40), "14 Hz\nFrame",  14.0, FlickerMode.FRAME_COUNT,   0.00),
        ((200, 200, 200), "15 Hz\nFrame",  15.0, FlickerMode.FRAME_COUNT,   0.00),
    ]

    paths = [_make_placeholder(c, lbl.replace("\n", " ")) for c, lbl, *_ in specs]

    positions = [
        (W * 1 // 5, H // 4), (W * 2 // 5, H // 4),
        (W * 3 // 5, H // 4), (W * 4 // 5, H // 4),
        (W * 1 // 5, H * 3 // 4), (W * 2 // 5, H * 3 // 4),
        (W * 3 // 5, H * 3 // 4), (W * 4 // 5, H * 3 // 4),
    ]

    return [
        Stimulus(
            image_path="./Images/WhiteSquare1.png",
            position=positions[i],
            size=(160, 160),
            target_freq=specs[i][2],
            mode=specs[i][3],
            phase=specs[i][4],
        )
        for i in range(8)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─── CONFIGURE YOUR EXPERIMENT HERE ──────────────────────────────────────
    #
    # Build your stimulus list and pass it to StimulusDisplay, e.g.:
    #
    #   STIMULI = [
    #       Stimulus(
    #           image_path = "checkerboard.png",
    #           position   = (W // 2, H // 2),
    #           size       = (200, 200),
    #           target_freq= 10.0,
    #           mode       = FlickerMode.APPROXIMATION,
    #           phase      = 0.0,    # 0.0 = starts ON; 0.5 = starts OFF
    #       ),
    #       Stimulus(
    #           image_path = "face.png",
    #           position   = (W // 4, H // 2),
    #           size       = (150, 150),
    #           target_freq= 12.0,
    #           mode       = FlickerMode.FRAME_COUNT,
    #           # phase is ignored in FRAME_COUNT mode
    #       ),
    #   ]
    #
    # ─── FREQUENCY GUIDANCE ──────────────────────────────────────────────────
    #
    # APPROXIMATION: any target_freq < refresh_rate / 2 works exactly.
    #
    # FRAME_COUNT:   only frequencies where refresh_rate / (2 * f) is an
    # integer are exact.  At 60 Hz these are 30, 15, 12, 10, 7.5, 6, 5, …
    # A startup warning is printed when deviation > 1 %.
    #
    # ─────────────────────────────────────────────────────────────────────────

    TARGET_FPS  = 60    # set to your monitor's native refresh rate
    FULLSCREEN  = True  # False for a 1280×720 windowed mode

    pygame.init()

    # Open the display FIRST so display.Info() returns the real screen size,
    # which we need to position stimuli across the full screen.
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF if FULLSCREEN else 0
    if FULLSCREEN:
        pygame.display.set_mode((0, 0), flags)
    else:
        pygame.display.set_mode((1280, 720), flags)

    info = pygame.display.Info()
    W, H = info.current_w, info.current_h

    STIMULI = create_demo_stimuli(W, H)

    StimulusDisplay(
        stimuli   = STIMULI,
        fullscreen= FULLSCREEN,
        fps       = TARGET_FPS,
        bg_color  = (0, 0, 0),
    ).run()