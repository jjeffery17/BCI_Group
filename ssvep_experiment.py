"""
ssvep_experiment.py
===================
SSVEP stimulus experiment controller using blinking_stimuli.py and PsychoPy.

Experiment structure
────────────────────
  4 rounds, one per target stimulus location:
    - Top-left      (6 Hz)
    - Top-right     (8 Hz)
    - Bottom-left   (11 Hz)
    - Bottom-right  (15 Hz)

  Each round:
    - Instruction screen: which light to focus on (press SPACE to start)
    - 5 continuous 5-second SSVEP epochs
    - A short, defined break between repeats

  Total: 4 rounds × 5 repeats × 5 s of stimulus time, plus short breaks.

Typical usage
─────────────
  1. Edit the EXPERIMENT CONFIG block near the bottom of this file.
  2. Run:  python ssvep_experiment.py
  3. A fullscreen PsychoPy window opens; follow the on-screen instructions.
  4. After the experiment a CSV event log is written alongside this file.

Dependencies
────────────
  pip install psychopy pygame numpy

  blinking_stimuli.py must be in the same directory as this file.

Notes on pygame / PsychoPy co-existence
────────────────────────────────────────
  PsychoPy opens its own OpenGL/pyglet window.  pygame is imported by
  blinking_stimuli.py but its display subsystem is never initialised here —
  only the Stimulus scheduling maths and image-processing helpers are used.
  pygame.display.init() is intentionally NOT called in this file so that
  PsychoPy's window remains the sole display owner.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Graceful optional imports
# ──────────────────────────────────────────────────────────────────────────────

# PsychoPy ────────────────────────────────────────────────────────────────────
try:
    from psychopy import core, event, visual, monitors
    from psychopy.hardware import keyboard
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False
    print(
        "[ssvep_experiment] WARNING: PsychoPy is not installed.\n"
        "  Install with:  pip install psychopy\n"
        "  The experiment cannot run without PsychoPy."
    )

# blinking_stimuli ────────────────────────────────────────────────────────────
try:
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import pygame  # imported transitively; we will NOT call pygame.display.init
    from blinking_stimuli import (
        Stimulus,
        FlickerMode,
        FlickerType,
        ColorMode,
        QuickLayout,
    )
    BLINKING_STIMULI_AVAILABLE = True
except ImportError as _err:
    BLINKING_STIMULI_AVAILABLE = False
    print(
        f"[ssvep_experiment] WARNING: Could not import blinking_stimuli.py "
        f"({_err}).\n"
        f"  Make sure blinking_stimuli.py is in the same directory as this file\n"
        f"  and that pygame is installed:  pip install pygame"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """All experiment parameters in one place.  Edit this block to configure."""

    # ── Display ───────────────────────────────────────────────────────────────
    monitor_name: str       = "testMonitor"   # PsychoPy monitor profile name
    screen_index: int       = 0               # 0 = primary display
    fullscreen: bool        = True
    background_color: tuple = (-1, -1, -1)    # PsychoPy RGB in [-1, 1]; black
    target_fps: int         = 75              # Must match monitor native refresh rate

    # ── Stimuli ───────────────────────────────────────────────────────────────
    image_paths: list       = field(default_factory=lambda: ["placeholder.png"])
    stimulus_size: tuple    = (150, 150)      # pixels (width, height)
    flicker_frequencies: list = field(
        default_factory=lambda: [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    )
    flicker_mode: object    = None            # set in __post_init__
    flicker_type: object    = None            # set in __post_init__
    color_mode: object      = None            # set in __post_init__

    # Layout: "grid" | "circle" | "checkerboard"
    layout: str             = "grid"
    layout_rows: int        = 2
    layout_cols: int        = 4
    layout_circle_n: int    = 8
    layout_circle_radius: float = 0.38

    # ── Timing ────────────────────────────────────────────────────────────────
    # Structure: 4 rounds × n_repeats_per_round × trial_duration_s
    n_rounds: int           = 4     # one round per target location
    n_repeats_per_round: int = 5    # SSVEP epochs within each round
    trial_duration_s: float = 5.0   # seconds per SSVEP epoch
    repeat_break_s: float   = 0.75  # short pause between repeats
    pre_experiment_rest_s: float = 2.0

    # ── Round labels ─────────────────────────────────────────────────────────
    # Order must match the active stimulus positions in your layout.
    # Default: top-left, top-right, bottom-left, bottom-right.
    round_labels: list      = field(default_factory=lambda: [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
    ])

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "."
    log_filename: str = "ssvep_events_{datetime}.csv"

    def __post_init__(self):
        if self.flicker_mode is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.flicker_mode = FlickerMode.APPROXIMATION
        if self.flicker_type is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.flicker_type = FlickerType.ON_OFF
        if self.color_mode is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.color_mode = ColorMode.GREYSCALE


@dataclass
class EventMarker:
    """A single time-stamped event written to the CSV log."""
    event_type: str          # e.g. "EXPERIMENT_START", "ROUND_START", "EPOCH_START"
    psychopy_time_s: float
    round_number: int        = -1   # 1-based
    repeat_number: int       = -1   # 1-based within round
    round_label: str         = ""   # e.g. "top left"
    notes: str               = ""


class ExperimentAborted(RuntimeError):
    """Raised when the user aborts the experiment via keyboard or window close."""


# ──────────────────────────────────────────────────────────────────────────────
# PsychoPy stimulus wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PsychoPyStimDriver:
    """
    Drives blinking_stimuli Stimulus objects inside a PsychoPy Window.

    PsychoPy owns the display flip cycle.  This class:
      - Wraps a list of blinking_stimuli.Stimulus objects.
      - Converts pygame Surfaces to PsychoPy ImageStim objects once at startup
        (using numpy as an intermediary — no pygame display window is opened).
      - On every frame: checks each stimulus's _on_phase, blits the correct
        PsychoPy texture, then calls stimulus.update() after the flip.

    Coordinate mapping
    ──────────────────
    blinking_stimuli uses pixel coordinates (origin top-left).
    PsychoPy uses the 'pix' unit system (origin centre, y up).
    Conversion: px_x → px_x - W/2,  px_y → -(px_y - H/2)
    """

    def __init__(
        self,
        win: "visual.Window",
        stimuli: list,
        refresh_rate: float,
    ):
        self._win          = win
        self._stimuli      = stimuli
        self._refresh_rate = refresh_rate
        self._psychopy_on:  list = []
        self._psychopy_off: list = []

        W, H = win.size
        self._W, self._H = W, H

        for s in stimuli:
            s._configure(refresh_rate)

        try:
            import numpy as np
            for s in stimuli:
                cx, cy = self._to_psychopy_coords(s.position)
                w, h   = s._image_on.get_size()

                on_arr  = self._surface_to_array(s._image_on)
                on_stim = visual.ImageStim(
                    win, image=on_arr, pos=(cx, cy), size=(w, h), units="pix"
                )
                self._psychopy_on.append(on_stim)

                if s._image_off is not None:
                    off_arr  = self._surface_to_array(s._image_off)
                    off_stim = visual.ImageStim(
                        win, image=off_arr, pos=(cx, cy), size=(w, h), units="pix"
                    )
                    self._psychopy_off.append(off_stim)
                else:
                    self._psychopy_off.append(None)

        except ImportError:
            raise RuntimeError(
                "numpy is required for PsychoPyStimDriver surface conversion.\n"
                "  pip install numpy"
            )

    def draw(self) -> None:
        """Draw all stimuli for the current frame (call before win.flip())."""
        for i, s in enumerate(self._stimuli):
            if s._on_phase:
                self._psychopy_on[i].draw()
            elif self._psychopy_off[i] is not None:
                self._psychopy_off[i].draw()

    def update(self) -> None:
        """Advance all stimulus frame counters (call after win.flip())."""
        for s in self._stimuli:
            s.update()

    def reset_frames(self) -> None:
        """Reset all frame counters to 0 (call at epoch onset)."""
        for s in self._stimuli:
            s._frame = 0

    def _to_psychopy_coords(self, pos: tuple) -> tuple:
        px, py = pos
        return (px - self._W / 2, -(py - self._H / 2))

    @staticmethod
    def _surface_to_array(surf: "pygame.Surface"):
        """
        Convert a pygame Surface to a numpy float32 array for PsychoPy ImageStim.
        pygame.surfarray.array3d → (W, H, 3); we transpose to (H, W, 3) then
        normalise to [-1, 1].
        """
        import numpy as np
        import pygame
        import tempfile
        import time
        from pathlib import Path

        arr   = pygame.surfarray.array3d(surf)   # (W, H, 3)
        arr   = arr.transpose(1, 0, 2)           # (H, W, 3)
        arr_f = arr.astype(np.float32) / 127.5 - 1.0

        try:
            mn   = float(arr_f.min())
            mx   = float(arr_f.max())
            mean = float(arr_f.mean())
        except Exception:
            mn = mx = mean = None
        print(
            f"[PsychoPyStimDriver] _surface_to_array: shape={arr_f.shape} "
            f"dtype={arr_f.dtype} min={mn:.3f} max={mx:.3f} mean={mean:.3f}",
            flush=True,
        )

        try:
            from PIL import Image
            tmpdir = Path(tempfile.gettempdir()) / "ssvep_debug"
            tmpdir.mkdir(exist_ok=True)
            preview = ((arr_f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            fname   = tmpdir / f"stim_preview_{int(time.time()*1000)}.png"
            Image.fromarray(preview).save(str(fname))
            print(f"[PsychoPyStimDriver] saved preview → {fname}", flush=True)
        except Exception:
            pass

        return arr_f


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment class
# ──────────────────────────────────────────────────────────────────────────────

class SSVEPExperiment:
    """
    Orchestrates the full SSVEP experiment session.

    Structure
    ─────────
      For each of 4 rounds:
        1. Instruction screen naming the target (e.g. "top left").
           Participant presses SPACE to begin.
        2. 5 consecutive 5-second SSVEP epochs with a short break between repeats.

      Total stimulus time: 4 × 5 × 5 s, plus the repeat breaks.

    Usage
    ─────
      SSVEPExperiment(config).run_full_session()
    """

    def __init__(self, config: ExperimentConfig):
        self._cfg       = config
        self._win       = None
        self._clock     = None
        self._stimuli   = []
        self._driver    = None
        self._event_log: list[EventMarker] = []
        self._kb        = None

    # ── public entry point ────────────────────────────────────────────────────

    def run_full_session(self) -> None:
        """Run setup → experiment → teardown in sequence."""
        print("[SSVEPExperiment] run_full_session: starting", flush=True)
        try:
            self.setup()
            self.run()
        except ExperimentAborted as exc:
            print(f"[SSVEPExperiment] Aborted: {exc}", flush=True)
        finally:
            self.teardown()
            print("[SSVEPExperiment] run_full_session: finished", flush=True)

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialise display and stimuli."""
        print("[SSVEPExperiment] setup: starting", flush=True)
        self._check_dependencies()
        print("[SSVEPExperiment] setup: dependencies OK", flush=True)

        # ── PsychoPy window ───────────────────────────────────────────────────
        mon = monitors.Monitor(self._cfg.monitor_name)
        print("[SSVEPExperiment] setup: creating PsychoPy window...", flush=True)
        try:
            self._win = visual.Window(
                monitor=mon,
                screen=self._cfg.screen_index,
                fullscr=self._cfg.fullscreen,
                color=self._cfg.background_color,
                colorSpace="rgb",
                units="pix",
                winType="pyglet",
                allowGUI=False,
                checkTiming=False,
            )
            print("[SSVEPExperiment] setup: PsychoPy window created", flush=True)
        except Exception as exc:
            import traceback
            print(f"[SSVEPExperiment] ERROR creating PsychoPy window: {exc}", flush=True)
            traceback.print_exc()
            raise

        W, H = self._win.size

        # ── status overlay ────────────────────────────────────────────────────
        try:
            panel_w = int(W * 0.6)
            panel_h = int(H * 0.22)
            panel_x = -W / 2 + panel_w / 2 + 20
            panel_y =  H / 2 - panel_h / 2 - 20
            self._status_box = visual.Rect(
                self._win,
                width=panel_w,
                height=panel_h,
                units="pix",
                pos=(panel_x, panel_y),
                fillColor=(-0.2, -0.2, -0.2),
                colorSpace="rgb",
                lineColor=None,
                opacity=0.85,
            )
            self._status_stim = visual.TextStim(
                self._win,
                text="",
                color=(1, 1, 1),
                height=20,
                units="pix",
                pos=(panel_x - panel_w / 2 + 10, panel_y + panel_h / 2 - 10),
                alignText="left",
                anchorHoriz="left",
                anchorVert="top",
                wrapWidth=panel_w - 20,
                font="Courier New",
            )
        except Exception:
            self._status_box  = None
            self._status_stim = visual.TextStim(
                self._win, text="", color=(1, 1, 1), height=22, units="pix"
            )

        # ── measure refresh rate ──────────────────────────────────────────────
        try:
            print("[SSVEPExperiment] Measuring frame rate interactively", flush=True)
            measured = self._measure_refresh_rate_interactive()
        except ExperimentAborted:
            raise
        except Exception as exc:
            measured = None
            import traceback
            print(f"[SSVEPExperiment] refresh-rate measurement failed: {exc}", flush=True)
            traceback.print_exc()

        refresh_rate = measured if (measured and measured > 10) else float(self._cfg.target_fps)
        measured_str = f"{measured:.2f}" if measured else "N/A"
        print(
            f"[SSVEPExperiment] PsychoPy measured frame rate: {measured_str} Hz  "
            f"(using {refresh_rate:.1f} Hz)",
            flush=True,
        )
        if abs(refresh_rate - self._cfg.target_fps) > 2.0:
            print(
                f"[SSVEPExperiment] WARNING: Measured rate ({refresh_rate:.1f} Hz) "
                f"differs from target_fps ({self._cfg.target_fps}). "
                f"Update target_fps in ExperimentConfig."
            )

        self._refresh_rate = refresh_rate
        self._clock = core.Clock()

        # ── keyboard ─────────────────────────────────────────────────────────
        self._kb = keyboard.Keyboard()

        # ── build stimuli ─────────────────────────────────────────────────────
        print(f"[SSVEPExperiment] Building stimuli for W={W}, H={H}", flush=True)
        self._stimuli = self._build_stimuli(W, H)
        print(
            f"[SSVEPExperiment] {len(self._stimuli)} stimuli created "
            f"({self._cfg.layout} layout)."
        )

        # ── PsychoPy stimulus driver ──────────────────────────────────────────
        print("[SSVEPExperiment] Initialising PsychoPyStimDriver", flush=True)
        self._driver = PsychoPyStimDriver(self._win, self._stimuli, refresh_rate)
        print("[SSVEPExperiment] PsychoPyStimDriver initialised", flush=True)

    # ── main experiment ───────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the full experiment:
          4 rounds × 5 repeats × 5-second SSVEP epochs.

        Between rounds an instruction screen names the target stimulus and
        waits for SPACE before proceeding.  A short, configurable break is
        inserted between repeats within a round.
        """
        cfg   = self._cfg
        clock = self._clock

        clock.reset()
        self._log_event(EventMarker(
            event_type="EXPERIMENT_START",
            psychopy_time_s=clock.getTime(),
            notes=f"rounds={cfg.n_rounds} repeats={cfg.n_repeats_per_round} "
                  f"epoch_s={cfg.trial_duration_s}",
        ))

        # brief preparation pause
        self._show_message("Preparing…")
        core.wait(cfg.pre_experiment_rest_s)

        labels = cfg.round_labels
        if len(labels) < cfg.n_rounds:
            # pad with generic labels if fewer labels than rounds were supplied
            labels = list(labels) + [
                f"stimulus {i+1}" for i in range(len(labels), cfg.n_rounds)
            ]

        # ── round loop ────────────────────────────────────────────────────────
        for round_idx in range(cfg.n_rounds):
            if self._check_quit():
                raise ExperimentAborted("user aborted before round")

            label = labels[round_idx]

            # ── inter-round instruction screen ────────────────────────────────
            instruction = (
                f"Round {round_idx + 1} of {cfg.n_rounds}\n\n"
                f"Please focus on the  {label.upper()}  flickering light.\n\n"
                f"Keep your gaze fixed on it throughout the round.\n\n"
                f"There will be five 5 second rounds with short gaps in between\n\n"
                f"Keep your gaze in the place where the light will be\n\n"
                f"Press  SPACE  to begin."
            )
            self._show_instruction(instruction)

            self._log_event(EventMarker(
                event_type="ROUND_START",
                psychopy_time_s=clock.getTime(),
                round_number=round_idx + 1,
                round_label=label,
                notes=f"repeats={cfg.n_repeats_per_round}",
            ))
            print(
                f"\n[SSVEPExperiment] ── Round {round_idx + 1}/{cfg.n_rounds}  "
                f"target={label!r} ──",
                flush=True,
            )

            # ── repeat loop ───────────────────────────────────────────────────
            for repeat_idx in range(cfg.n_repeats_per_round):
                if self._check_quit():
                    raise ExperimentAborted("user aborted during round")

                self._run_epoch(round_idx, repeat_idx, label)

                if repeat_idx < cfg.n_repeats_per_round - 1:
                    self._run_repeat_break(round_idx, repeat_idx, label)

            self._log_event(EventMarker(
                event_type="ROUND_END",
                psychopy_time_s=clock.getTime(),
                round_number=round_idx + 1,
                round_label=label,
            ))

        # ── end ───────────────────────────────────────────────────────────────
        self._log_event(EventMarker(
            event_type="EXPERIMENT_END",
            psychopy_time_s=clock.getTime(),
            notes=f"total_rounds={cfg.n_rounds}",
        ))

        self._show_message("Experiment complete.\n\nThank you!")
        core.wait(3.0)

        self._save_log()

    def _run_epoch(self, round_idx: int, repeat_idx: int, label: str) -> None:
        """Present all stimuli for one 5-second SSVEP epoch."""
        cfg   = self._cfg
        win   = self._win
        clock = self._clock

        # Reset blinking_stimuli frame counters so stimuli start in phase
        self._driver.reset_frames()

        epoch_onset = clock.getTime()
        self._log_event(EventMarker(
            event_type="EPOCH_START",
            psychopy_time_s=epoch_onset,
            round_number=round_idx + 1,
            repeat_number=repeat_idx + 1,
            round_label=label,
        ))

        # ── stimulus presentation loop ────────────────────────────────────────
        while clock.getTime() - epoch_onset < cfg.trial_duration_s:
            if self._check_quit():
                raise ExperimentAborted("user aborted during epoch")

            win.clearBuffer()
            self._driver.draw()
            win.flip()
            self._driver.update()

        epoch_end = clock.getTime()
        self._log_event(EventMarker(
            event_type="EPOCH_END",
            psychopy_time_s=epoch_end,
            round_number=round_idx + 1,
            repeat_number=repeat_idx + 1,
            round_label=label,
            notes=f"duration_s={epoch_end - epoch_onset:.4f}",
        ))

        print(
            f"  Round {round_idx + 1}  Repeat {repeat_idx + 1}/{cfg.n_repeats_per_round}  "
            f"t={epoch_onset:.3f}s  dur={epoch_end - epoch_onset:.3f}s",
            flush=True,
        )

    def _run_repeat_break(self, round_idx: int, repeat_idx: int, label: str) -> None:
        """Show a short blank rest period between repeats."""
        cfg = self._cfg
        clock = self._clock
        break_start = clock.getTime()

        self._log_event(EventMarker(
            event_type="REPEAT_BREAK_START",
            psychopy_time_s=break_start,
            round_number=round_idx + 1,
            repeat_number=repeat_idx + 1,
            round_label=label,
            notes=f"break_s={cfg.repeat_break_s:.3f}",
        ))

        core.wait(cfg.repeat_break_s)

        break_end = clock.getTime()
        self._log_event(EventMarker(
            event_type="REPEAT_BREAK_END",
            psychopy_time_s=break_end,
            round_number=round_idx + 1,
            repeat_number=repeat_idx + 1,
            round_label=label,
            notes=f"duration_s={break_end - break_start:.4f}",
        ))

    # ── teardown ──────────────────────────────────────────────────────────────

    def teardown(self) -> None:
        """Close the PsychoPy window."""
        if self._win:
            try:
                self._win.close()
            except Exception as exc:
                print(f"[SSVEPExperiment] Window close error: {exc}", flush=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _show_message(self, text: str) -> None:
        """Display a centred text message for one frame."""
        msg = visual.TextStim(
            self._win,
            text=text,
            color=(1, 1, 1),
            height=30,
            units="pix",
            wrapWidth=self._win.size[0] * 0.8,
        )
        self._win.clearBuffer()
        msg.draw()
        self._win.flip()

    def _show_instruction(self, text: str) -> None:
        """
        Display an instruction screen and wait for the participant to press
        SPACE (or Escape/Q to abort).  The screen remains visible until a key
        is pressed.
        """
        msg = visual.TextStim(
            self._win,
            text=text,
            color=(1, 1, 1),
            height=32,
            units="pix",
            wrapWidth=self._win.size[0] * 0.75,
        )
        while True:
            if self._abort_requested():
                raise ExperimentAborted("user aborted on instruction screen")

            self._win.clearBuffer()
            msg.draw()
            self._win.flip()

            keys = event.getKeys(keyList=["space", "escape", "q"])
            if "space" in keys:
                break
            if "escape" in keys or "q" in keys:
                raise ExperimentAborted("user aborted on instruction screen")

    def _measure_refresh_rate_interactive(
        self,
        min_seconds: float = 1.0,
        max_seconds: float = 3.0,
        target_samples: int = 90,
    ) -> float:
        """Measure refresh rate without blocking the UI."""
        import statistics

        samples: list[float] = []
        last_flip = None
        start     = time.perf_counter()
        warmup    = 10

        while True:
            elapsed = time.perf_counter() - start
            if self._abort_requested():
                raise ExperimentAborted("user aborted during refresh-rate measurement")

            self._update_status([
                "Detecting refresh rate...",
                f"Collected samples: {len(samples)}/{target_samples}",
                f"Elapsed: {elapsed:.1f}s / {max_seconds:.1f}s",
                "Press Esc or close the window to abort",
            ])

            self._win.clearBuffer()
            if hasattr(self, "_status_box") and self._status_box is not None:
                try:
                    self._status_box.draw()
                except Exception:
                    pass
            if hasattr(self, "_status_stim") and self._status_stim is not None:
                self._status_stim.draw()
            self._win.flip()

            flip_time = time.perf_counter()
            if last_flip is not None and warmup <= 0:
                samples.append(flip_time - last_flip)
            last_flip  = flip_time
            warmup    -= 1

            if elapsed >= min_seconds and len(samples) >= target_samples:
                break
            if elapsed >= max_seconds:
                break

        if not samples:
            return float(self._cfg.target_fps)

        median_interval = statistics.median(samples)
        if median_interval <= 0:
            return float(self._cfg.target_fps)
        return 1.0 / median_interval

    def _build_stimuli(self, W: int, H: int) -> list:
        """Construct the stimulus list using blinking_stimuli.QuickLayout."""
        cfg    = self._cfg
        layout = cfg.layout.lower()

        try:
            image_paths = self._resolve_image_paths(
                cfg.image_paths, len(cfg.flicker_frequencies)
            )
        except BaseException as exc:
            import traceback
            print(f"[SSVEPExperiment] ERROR resolving image paths: {exc}", flush=True)
            traceback.print_exc()
            raise

        try:
            if layout == "grid":
                print("[SSVEPExperiment] Creating grid layout via QuickLayout.grid", flush=True)
                return QuickLayout.grid(
                    image_paths  = image_paths,
                    W=W, H=H,
                    rows         = cfg.layout_rows,
                    cols         = cfg.layout_cols,
                    size         = cfg.stimulus_size,
                    target_freq  = cfg.flicker_frequencies,
                    flicker_mode = cfg.flicker_mode,
                    flicker_type = cfg.flicker_type,
                    color_mode   = cfg.color_mode,
                )
            elif layout == "circle":
                print("[SSVEPExperiment] Creating circle layout via QuickLayout.circle", flush=True)
                return QuickLayout.circle(
                    image_paths  = image_paths,
                    W=W, H=H,
                    n            = cfg.layout_circle_n,
                    radius       = cfg.layout_circle_radius,
                    size         = cfg.stimulus_size,
                    target_freq  = cfg.flicker_frequencies,
                    flicker_mode = cfg.flicker_mode,
                    flicker_type = cfg.flicker_type,
                    color_mode   = cfg.color_mode,
                )
            elif layout == "checkerboard":
                print("[SSVEPExperiment] Creating checkerboard layout", flush=True)
                rows   = cfg.layout_rows
                cols   = cfg.layout_cols
                n_cells = rows * cols

                if len(image_paths) < 2:
                    raise ValueError(
                        "Checkerboard layout requires at least 2 image paths (A and B)"
                    )
                img_a = image_paths[0]
                img_b = image_paths[1]
                freqs = cfg.flicker_frequencies

                if isinstance(freqs, (list, tuple)) and len(freqs) == 2:
                    tf_a, tf_b = freqs
                elif isinstance(freqs, (list, tuple)) and len(freqs) == n_cells:
                    tf_a, tf_b = [], []
                    for i in range(n_cells):
                        r = i // cols
                        c = i % cols
                        if (r + c) % 2 == 0:
                            tf_a.append(freqs[i])
                        else:
                            tf_b.append(freqs[i])
                else:
                    raise ValueError(
                        f"checkerboard: flicker_frequencies must have 2 values or "
                        f"{n_cells} (one per cell); got {len(freqs)}."
                    )

                return QuickLayout.checkerboard(
                    image_path_a = img_a,
                    image_path_b = img_b,
                    W=W, H=H,
                    rows         = rows,
                    cols         = cols,
                    size         = cfg.stimulus_size,
                    target_freq_a = tf_a,
                    target_freq_b = tf_b,
                    flicker_mode = cfg.flicker_mode,
                    flicker_type = cfg.flicker_type,
                    color_mode   = cfg.color_mode,
                )
            else:
                raise ValueError(
                    f"Unknown layout '{cfg.layout}'. "
                    "Choose 'grid', 'circle', or 'checkerboard'."
                )
        except Exception as exc:
            import traceback
            print(f"[SSVEPExperiment] ERROR building stimuli: {exc}", flush=True)
            traceback.print_exc()
            raise

    @staticmethod
    def _resolve_image_paths(paths: list, n_stimuli: int) -> list:
        """Return n_stimuli image paths, generating colour placeholders as needed."""
        from blinking_stimuli import _make_placeholder

        PLACEHOLDER_COLOURS = [
            (220, 60, 60), (60, 180, 60), (60, 120, 220), (200, 160, 40),
            (160, 60, 210), (40, 200, 190), (220, 120, 40), (200, 200, 200),
        ]
        resolved = []
        for i in range(n_stimuli):
            p = paths[i % len(paths)]
            if Path(p).exists():
                resolved.append(p)
            else:
                colour = PLACEHOLDER_COLOURS[i % len(PLACEHOLDER_COLOURS)]
                resolved.append(_make_placeholder(colour, f"S{i+1}"))
        return resolved

    def _log_event(self, marker: EventMarker) -> None:
        self._event_log.append(marker)

    def _save_log(self) -> None:
        """Write the event log to a CSV file."""
        if not self._event_log:
            return
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn     = self._cfg.log_filename.replace("{datetime}", ts)
        out    = Path(self._cfg.output_dir) / fn
        fields = list(asdict(self._event_log[0]).keys())
        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for marker in self._event_log:
                    writer.writerow(asdict(marker))
            print(f"[SSVEPExperiment] Event log saved → {out}")
        except Exception as exc:
            print(f"[SSVEPExperiment] Failed to save log: {exc}")

    def _update_status(self, lines: list) -> None:
        try:
            if not hasattr(self, "_status_stim") or self._status_stim is None:
                return
            self._status_stim.text = "\n".join(lines)
        except Exception:
            pass

    def _abort_requested(self) -> bool:
        try:
            keys = event.getKeys(keyList=["escape", "q"])
            if keys:
                return True
        except Exception:
            pass
        try:
            if self._win is not None and getattr(self._win, "winHandle", None) is not None:
                self._win.winHandle.dispatch_events()
                if getattr(self._win.winHandle, "has_exit", False):
                    return True
        except Exception:
            pass
        return False

    def _check_quit(self) -> bool:
        return self._abort_requested()

    @staticmethod
    def _check_dependencies() -> None:
        missing = []
        if not PSYCHOPY_AVAILABLE:
            missing.append("psychopy  (pip install psychopy)")
        if not BLINKING_STIMULI_AVAILABLE:
            missing.append("blinking_stimuli.py  (place in same directory)")
        if missing:
            raise RuntimeError(
                "Missing required dependencies:\n  " + "\n  ".join(missing)
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIGURE YOUR EXPERIMENT HERE ───────────────────────────────────────
    config = ExperimentConfig(
        # ── Display ────────────────────────────────────────────────────────
        monitor_name  = "testMonitor",
        screen_index  = 0,
        fullscreen    = True,
        target_fps    = 75,          # ← set to your monitor's refresh rate

        # ── Stimuli ────────────────────────────────────────────────────────
        # Images cycle across the 8 grid positions (white/black alternating).
        image_paths = [
            "Images/WhiteSquare1.png", "Images/BlackSquare1.png",
            "Images/WhiteSquare1.png", "Images/BlackSquare1.png",
            "Images/BlackSquare1.png", "Images/WhiteSquare1.png",
            "Images/BlackSquare1.png", "Images/WhiteSquare1.png",
        ],
        stimulus_size       = (150, 150),
        # Active stimuli (non-zero frequency) sit at the four corners of the
        # 2×4 grid:  top-left=6 Hz, top-right=8 Hz,
        #            bottom-left=11 Hz, bottom-right=15 Hz.
        # Zero-frequency entries are static (no flicker).
        flicker_frequencies = [
            6.0,  0.0,  8.0,  0.0,   # row 0:  TL  --  TR  --
            0.0, 11.0,  0.0, 15.0,   # row 1:  --  BL  --  BR
        ],
        layout      = "grid",
        layout_rows = 2,
        layout_cols = 4,

        # ── Timing ─────────────────────────────────────────────────────────
        n_rounds           = 4,   # one round per target location
        n_repeats_per_round = 5,  # epochs per round
        trial_duration_s   = 5.0, # seconds per epoch
        repeat_break_s     = 0.75, # short pause between repeats
        pre_experiment_rest_s = 2.0,

        # ── Round labels ───────────────────────────────────────────────────
        # Must match the order of your active stimuli (left→right, top→bottom).
        round_labels = [
            "top left",
            "top right",
            "bottom left",
            "bottom right",
        ],

        # ── Output ─────────────────────────────────────────────────────────
        output_dir   = ".",
        log_filename = "ssvep_events_{datetime}.csv",
    )
    # ─────────────────────────────────────────────────────────────────────────

    try:
        print("[main] Starting SSVEPExperiment", flush=True)
        SSVEPExperiment(config).run_full_session()
    except ExperimentAborted as exc:
        print(f"[main] Experiment aborted cleanly: {exc}", flush=True)
    except BaseException as exc:
        import traceback
        print(f"[main] Unhandled exception (type={type(exc).__name__}): {exc}", flush=True)
        traceback.print_exc()
        raise