"""
ssvep_experiment.py
===================
Experiment controller that bridges blinking_stimuli.py, PsychoPy, and the
IDUN Guardian EEG earbud SDK.

Architecture overview
─────────────────────
  Main thread   — PsychoPy Window + blinking_stimuli Stimulus objects.
                  Owns the display flip cycle and all trial timing.
                  Uses PsychoPy's core.Clock as the single timing authority.

  Background thread — IDUN GuardianClient running its own asyncio event loop.
                  Starts/stops the EEG recording and (optionally) subscribes
                  to real-time insights without blocking the stimulus loop.

  Thread-safe queue — carries EventMarker objects from the main thread to the
                  background thread so that stimulus onsets are time-stamped
                  against the IDUN cloud recording ID for later alignment.

Typical usage
─────────────
  1. Edit the EXPERIMENT CONFIG block near the bottom of this file.
  2. Run:  python ssvep_experiment.py
  3. The console prints impedance / battery info, then the trial loop starts.
  4. After the experiment the IDUN recording is stopped, data downloaded, and
     a CSV event log is written alongside this file.

Dependencies
────────────
  Required:
    pip install psychopy pygame

  For IDUN data collection (optional — experiment runs without it):
    pip install idun-guardian-sdk
    Set IDUN_API_TOKEN environment variable or fill in IDUN_API_TOKEN below.

  blinking_stimuli.py must be in the same directory as this file.

Notes on pygame / PsychoPy co-existence
─────────────────────────────────────────
  PsychoPy opens its own OpenGL/pyglet window.  pygame is imported by
  blinking_stimuli.py but its display subsystem is never initialised here —
  only the Stimulus scheduling maths and image-processing helpers are used.
  pygame.display.init() is intentionally NOT called in this file so that
  PsychoPy's window remains the sole display owner.
"""

from __future__ import annotations

import asyncio
import csv
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


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
    # Allow blinking_stimuli.py to live alongside this file without being on
    # sys.path by default.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    # We import only the scheduling / image classes — NOT the StimulusDisplay
    # application class, which owns its own pygame display loop.
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

# IDUN Guardian SDK ────────────────────────────────────────────────────────────
try:
    from idun_guardian_sdk import GuardianClient, FileTypes
    IDUN_AVAILABLE = True
except ImportError:
    IDUN_AVAILABLE = False
    print(
        "[ssvep_experiment] INFO: idun-guardian-sdk is not installed.\n"
        "  The experiment will run without EEG recording.\n"
        "  Install with:  pip install idun-guardian-sdk"
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
    background_color: tuple = (-1, -1, -1)    # PsychoPy RGB in [-1, 1]; (-1,-1,-1) = black
    target_fps: int         = 60              # Must match monitor native refresh rate

    # ── Stimuli ───────────────────────────────────────────────────────────────
    # image_paths: list of file paths for the stimulus images.
    # Use a single shared image or one per stimulus.  Cycled via QuickLayout.
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
    n_trials: int           = 20
    trial_duration_s: float = 4.0     # seconds each stimulus epoch runs for
    isi_duration_s: float   = 1.0     # inter-stimulus interval (blank screen)
    pre_experiment_rest_s: float = 2.0  # rest period before first trial

    # ── IDUN Guardian ────────────────────────────────────────────────────────
    enable_idun: bool           = True
    idun_api_token: str         = ""          # or set IDUN_API_TOKEN env var
    idun_device_address: str    = ""          # leave blank to auto-search
    idun_recording_duration_s: int = 3600     # 1 hour max; auto-stopped at end
    idun_stream_live_eeg: bool  = True        # subscribe to raw EEG live stream
    idun_mains_60hz: bool       = False       # True for North America (60 Hz)
    idun_led_sleep: bool        = False       # False = keep LED on during recording
    idun_impedance_check: bool  = True        # run impedance check before experiment
    idun_impedance_duration_s: int = 10       # seconds to stream impedance
    idun_download_after: bool   = True        # download EEG file when done

    # ── Output ────────────────────────────────────────────────────────────────
    # Event log will be written to this path.  {datetime} is substituted.
    output_dir: str = "."
    log_filename: str = "ssvep_events_{datetime}.csv"

    def __post_init__(self):
        if self.flicker_mode is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.flicker_mode = FlickerMode.APPROXIMATION
        if self.flicker_type is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.flicker_type = FlickerType.ON_NEGATIVE
        if self.color_mode is None:
            if BLINKING_STIMULI_AVAILABLE:
                self.color_mode = ColorMode.GREYSCALE


@dataclass
class EventMarker:
    """A single time-stamped event written to the CSV log.

    All times are in seconds relative to the experiment start clock
    (psychopy.core.Clock started at experiment onset).
    """
    event_type: str          # "TRIAL_START" | "TRIAL_END" | "ISI_START" | "EXPERIMENT_START" | "EXPERIMENT_END"
    psychopy_time_s: float   # PsychoPy clock time at this event
    trial_number: int        = -1
    stimulus_index: int      = -1    # index into stimuli list; -1 = N/A
    target_freq_hz: float    = -1.0
    actual_freq_hz: float    = -1.0
    idun_recording_id: str   = ""    # populated once IDUN recording starts
    notes: str               = ""


class ExperimentAborted(RuntimeError):
    """Raised when the user aborts the experiment via keyboard or window close."""



# ──────────────────────────────────────────────────────────────────────────────
# IDUN Guardian background recorder
# ──────────────────────────────────────────────────────────────────────────────

class IdunRecorder:
    """
    Wraps GuardianClient in a background thread with its own asyncio loop.

    The IDUN SDK requires all calls to share the same asyncio event loop (Bleak
    BLE limitation).  Running this in a background thread lets the main thread
    drive the PsychoPy stimulus display without interruption.

    Usage
    ─────
        recorder = IdunRecorder(config)
        recorder.start()                      # non-blocking; returns immediately
        recording_id = recorder.recording_id  # available once recording begins
        recorder.stop()                       # graceful shutdown
    """

    def __init__(self, config: ExperimentConfig):
        self._config       = config
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[GuardianClient] = None
        self._ready_event  = threading.Event()   # set when recording starts
        self._stop_event   = threading.Event()   # set to request shutdown
        self._recording_id: Optional[str] = None
        self._error: Optional[Exception]  = None
        self._live_handler: Optional[Callable] = None
        # Human-readable status string updated by the background thread.
        self._status_msg: str = "initialised"

        # Shared queue: main thread pushes EventMarker; background logs them
        self.marker_queue: queue.Queue[EventMarker] = queue.Queue()

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def recording_id(self) -> Optional[str]:
        return self._recording_id

    @property
    def is_ready(self) -> bool:
        """True once the IDUN recording has successfully started."""
        return self._ready_event.is_set()

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def set_live_eeg_handler(self, handler: Callable) -> None:
        """Optionally set a callback for raw EEG live insight packets."""
        self._live_handler = handler

    def start(self) -> None:
        """Start the background thread.  Returns immediately."""
        if not IDUN_AVAILABLE:
            print("[IdunRecorder] IDUN SDK not available — skipping EEG recording.")
            self._ready_event.set()   # unblock any waiters
            return
        self._status_msg = "starting background thread"
        self._thread = threading.Thread(
            target=self._run_loop, name="IdunRecorderThread", daemon=True
        )
        self._thread.start()

    def wait_until_ready(self, timeout_s: float = 60.0) -> bool:
        """Block the calling thread until the recording has started or timeout."""
        return self._ready_event.wait(timeout=timeout_s)

    def push_marker(self, marker: EventMarker) -> None:
        """Thread-safe: queue an event marker from the main thread."""
        if self._recording_id:
            marker.idun_recording_id = self._recording_id
        self.marker_queue.put_nowait(marker)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)

    # ── background thread ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            self._error = exc
            print(f"[IdunRecorder] Background thread error: {exc}")
        finally:
            self._loop.close()
            self._ready_event.set()   # unblock any waiters even on error

    async def _async_main(self) -> None:
        cfg = self._config

        api_token = cfg.idun_api_token or os.environ.get("IDUN_API_TOKEN", "")
        address   = cfg.idun_device_address or None

        self._client = GuardianClient(
            api_token=api_token,
            address=address,
        )

        # ── connect ───────────────────────────────────────────────────────────
        self._status_msg = "Connecting to IDUN Guardian earbud…"
        print("[IdunRecorder] Connecting to IDUN Guardian earbud…")
        try:
            await self._client.connect_device()
            self._status_msg = "Connected"
            print("[IdunRecorder] Connected.")
        except Exception as exc:
            self._status_msg = f"Connection failed: {exc}"
            print(f"[IdunRecorder] Connection failed: {exc}")
            raise

        # ── battery check ────────────────────────────────────────────────────
        try:
            self._status_msg = "Checking battery level..."
            battery = await self._client.check_battery()
            self._status_msg = f"Battery: {battery}%"
            print(f"[IdunRecorder] Battery level: {battery}%")
            if battery < 20:
                print("[IdunRecorder] WARNING: Battery below 20% — consider charging.")
        except Exception as exc:
            self._status_msg = f"Battery check failed: {exc}"
            print(f"[IdunRecorder] Battery check failed: {exc}")

        # ── impedance check ──────────────────────────────────────────────────
        if cfg.idun_impedance_check:
            self._status_msg = "Running impedance check..."
            await self._run_impedance_check()
            self._status_msg = "Impedance check complete"

        # ── subscribe to live EEG ────────────────────────────────────────────
        if cfg.idun_stream_live_eeg:
            handler = self._live_handler or self._default_eeg_handler
            self._status_msg = "Subscribing to live EEG insights..."
            self._client.subscribe_live_insights(
                raw_eeg=True,
                filtered_eeg=False,
                imu=False,
                handler=handler,
            )
            self._status_msg = "Subscribed to live EEG"

        # ── start recording ───────────────────────────────────────────────────
        self._status_msg = f"Starting recording ({cfg.idun_recording_duration_s}s max)"
        print(f"[IdunRecorder] Starting recording ({cfg.idun_recording_duration_s}s max)…")

        # Retrieve the recording ID once the recording is initialised.
        # start_recording() blocks until the recording ends, so we must
        # obtain the ID before it returns.  The SDK stores it internally;
        # we poll briefly after the call returns — but we also need to grab it
        # before the await completes.  We do this by launching start_recording
        # as a task and querying get_recording_id() once it starts.
        record_task = asyncio.create_task(
            self._client.start_recording(
                recording_timer=cfg.idun_recording_duration_s,
                led_sleep=cfg.idun_led_sleep,
            )
        )

        # Give the recording task a moment to initialise, then grab the ID.
        await asyncio.sleep(2.0)
        try:
            self._recording_id = self._client.get_recording_id()
            self._status_msg = f"Recording started (ID: {self._recording_id})"
            print(f"[IdunRecorder] Recording started. ID: {self._recording_id}")
        except Exception:
            self._recording_id = None
            self._status_msg = "Could not retrieve recording ID yet"
            print("[IdunRecorder] Could not retrieve recording ID yet.")

        self._ready_event.set()   # signal main thread that recording is live

        # ── drain marker queue while recording runs ───────────────────────────
        while not self._stop_event.is_set():
            try:
                _marker = self.marker_queue.get_nowait()
                # Markers are consumed here; the main thread also writes them to
                # CSV independently, so this loop exists for future extension
                # (e.g. sending annotations to the IDUN cloud API).
            except queue.Empty:
                pass
            await asyncio.sleep(0.05)

        # ── stop recording ───────────────────────────────────────────────────
        print("[IdunRecorder] Stopping recording…")
        record_task.cancel()
        try:
            await record_task
        except (asyncio.CancelledError, Exception):
            pass

        # ── download data ─────────────────────────────────────────────────────
        if cfg.idun_download_after and self._recording_id:
            await self._download_data()

        # ── disconnect ────────────────────────────────────────────────────────
        try:
            await self._client.disconnect_device()
            print("[IdunRecorder] Disconnected.")
        except Exception as exc:
            print(f"[IdunRecorder] Disconnect error: {exc}")

    async def _run_impedance_check(self) -> None:
        """Stream impedance for a fixed duration and print the result."""
        cfg = self._config
        print(
            f"[IdunRecorder] Running impedance check "
            f"({cfg.idun_impedance_duration_s}s)…"
        )
        impedance_values: list[float] = []

        def _imp_handler(data):
            try:
                impedance_values.append(float(data))
            except (TypeError, ValueError):
                pass

        imp_task = asyncio.create_task(
            self._client.stream_impedance(
                handler=_imp_handler,
                mains_freq_60hz=cfg.idun_mains_60hz,
            )
        )
        await asyncio.sleep(cfg.idun_impedance_duration_s)
        self._client.stop_impedance()
        try:
            await imp_task
        except Exception:
            pass

        if impedance_values:
            avg_kohm = sum(impedance_values) / len(impedance_values) / 1000
            print(f"[IdunRecorder] Average impedance: {avg_kohm:.1f} kΩ", end="  ")
            if avg_kohm > 300:
                print("⚠ HIGH — reposition earbud and clean ear canal")
            else:
                print("✓ acceptable (< 300 kΩ)")
        else:
            print("[IdunRecorder] No impedance values received.")

    async def _download_data(self) -> None:
        cfg = self._config
        print(f"[IdunRecorder] Downloading EEG data for recording {self._recording_id}…")
        try:
            self._client.download_file(
                recording_id=self._recording_id,
                file_type=FileTypes.EEG,
            )
            print(f"[IdunRecorder] EEG download complete.")
        except Exception as exc:
            print(f"[IdunRecorder] Download failed: {exc}")

    @staticmethod
    def _default_eeg_handler(event) -> None:
        """Minimal live EEG handler — prints packet count for diagnostics."""
        try:
            n = len(event.message.get("raw_eeg", []))
            print(f"\r[IdunRecorder] Live EEG: {n} samples received", end="", flush=True)
        except Exception:
            pass


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
        self._psychopy_on:  list = []   # ImageStim for ON phase
        self._psychopy_off: list = []   # ImageStim for OFF phase (or None)

        W, H = win.size
        self._W, self._H = W, H

        # Configure blinking_stimuli scheduling (normally done by StimulusDisplay)
        for s in stimuli:
            s._configure(refresh_rate)

        # Build PsychoPy ImageStim objects from the pre-rendered pygame surfaces
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

    # ── per-frame interface ───────────────────────────────────────────────────

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
        """Reset all frame counters to 0 (call at trial onset)."""
        for s in self._stimuli:
            s._frame = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_psychopy_coords(self, pos: tuple) -> tuple:
        """Convert pixel (x, y) from top-left origin to PsychoPy centre origin."""
        px, py = pos
        return (px - self._W / 2, -(py - self._H / 2))

    @staticmethod
    def _surface_to_array(surf: "pygame.Surface"):
        """
        Convert a pygame Surface to a numpy uint8 array suitable for
        psychopy.visual.ImageStim.

        PsychoPy's ImageStim accepts an (H, W, 3) uint8 array.
        pygame.surfarray.array3d returns (W, H, 3), so we transpose axes 0 and 1.
        """
        import numpy as np
        import pygame
        import tempfile
        import time
        from pathlib import Path

        arr = pygame.surfarray.array3d(surf)   # (W, H, 3)
        # Transpose to (H, W, 3)
        arr = arr.transpose(1, 0, 2)

        # Convert to float32 in range [-1, 1] as required by PsychoPy for
        # numpy texture arrays (black=-1, white=+1).
        arr_f = arr.astype(np.float32) / 127.5 - 1.0

        # Diagnostics: print concise array statistics (float range)
        try:
            mn = float(arr_f.min())
            mx = float(arr_f.max())
            mean = float(arr_f.mean())
        except Exception:
            mn = mx = mean = None
        print(
            f"[PsychoPyStimDriver] _surface_to_array: shape={arr_f.shape} dtype={arr_f.dtype} min={mn:.3f} max={mx:.3f} mean={mean:.3f}",
            flush=True,
        )

        # Save a uint8 preview for offline inspection (Pillow optional).
        try:
            from PIL import Image
            tmpdir = Path(tempfile.gettempdir()) / "ssvep_debug"
            tmpdir.mkdir(exist_ok=True)
            preview = ((arr_f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            fname = tmpdir / f"stim_preview_{int(time.time()*1000)}.png"
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

    Lifecycle
    ─────────
      experiment = SSVEPExperiment(config)
      experiment.setup()
      experiment.run()
      experiment.teardown()

    Or, equivalently:
      SSVEPExperiment(config).run_full_session()
    """

    def __init__(self, config: ExperimentConfig):
        self._cfg          = config
        self._win          = None
        self._clock        = None
        self._stimuli      = []
        self._driver       = None
        self._recorder     = None
        self._event_log: list[EventMarker] = []
        self._kb           = None

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
        """Initialise display, stimuli, IDUN recorder."""
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
                checkTiming=False,   # we do our own timing
            )
            print("[SSVEPExperiment] setup: PsychoPy window created", flush=True)
        except Exception as exc:
            import traceback
            print(f"[SSVEPExperiment] ERROR creating PsychoPy window: {exc}", flush=True)
            traceback.print_exc()
            raise
        W, H = self._win.size

        # Initialise a more visible status panel (semi-transparent box + monospace text)
        try:
            panel_w = int(W * 0.6)
            panel_h = int(H * 0.22)
            panel_x = -W / 2 + panel_w / 2 + 20
            panel_y = H / 2 - panel_h / 2 - 20
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
            self._status_box = None
            self._status_stim = visual.TextStim(self._win, text="", color=(1, 1, 1), height=22, units="pix")


        # Measure actual refresh rate using a non-blocking loop so the user can
        # see progress and abort with Escape / window close.
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
        print(f"[SSVEPExperiment] PsychoPy measured frame rate: {measured_str} Hz  (using {refresh_rate:.1f} Hz)", flush=True)
        if abs(refresh_rate - self._cfg.target_fps) > 2.0:
            print(
                f"[SSVEPExperiment] WARNING: Measured rate ({refresh_rate:.1f} Hz) "
                f"differs from TARGET_FPS ({self._cfg.target_fps}). "
                f"Update target_fps in ExperimentConfig."
            )

        self._refresh_rate = refresh_rate
        self._clock = core.Clock()

        # ── keyboard ─────────────────────────────────────────────────────────
        self._kb = keyboard.Keyboard()

        # ── build stimuli via blinking_stimuli QuickLayout ───────────────────
        print(f"[SSVEPExperiment] Building stimuli for W={W}, H={H}", flush=True)
        self._stimuli = self._build_stimuli(W, H)
        # Diagnostics: print image path and target frequency for each stimulus
        try:
            print("[SSVEPExperiment] Diagnostics: stimulus image paths and target freqs:", flush=True)
            for i, s in enumerate(self._stimuli):
                img = getattr(s, "image_path", None)
                tf  = getattr(s, "target_freq", None)
                fm  = getattr(s, "flicker_mode", None)
                ft  = getattr(s, "flicker_type", None)
                cm  = getattr(s, "color_mode", None)
                ph  = getattr(s, "phase", None)
                print(
                    f"  Stim {i}: image={img}  target_freq={tf}  "
                    f"flicker_mode={fm}  flicker_type={ft}  color_mode={cm}  phase={ph}",
                    flush=True,
                )
        except Exception:
            pass
        print(
            f"[SSVEPExperiment] {len(self._stimuli)} stimuli created "
            f"({self._cfg.layout} layout)."
        )

        # ── PsychoPy stimulus driver ─────────────────────────────────────────
        print("[SSVEPExperiment] Initialising PsychoPyStimDriver", flush=True)
        self._driver = PsychoPyStimDriver(self._win, self._stimuli, refresh_rate)
        print("[SSVEPExperiment] PsychoPyStimDriver initialised", flush=True)

        # ── IDUN recorder (background thread) ────────────────────────────────
        if self._cfg.enable_idun:
            self._recorder = IdunRecorder(self._cfg)
        else:
            # lightweight dummy to satisfy callers
            class _Dummy:
                is_ready = True
                error = None
                recording_id = None
                def start(self): pass
                def push_marker(self, m): pass
                def stop(self): pass
            self._recorder = _Dummy()
        print("[SSVEPExperiment] Starting IDUN recorder (if available)", flush=True)
        self._recorder.start()
        print("[SSVEPExperiment] IDUN recorder start() returned", flush=True)

        # Wait up to 90 s for the IDUN recording to start.  This covers the
        # impedance check + BLE connection.  We poll in a loop so we can
        # display progress on-screen (refresh-rate detection + connection).
        timeout = 90.0
        poll = 0.5
        start_t = time.time()
        print("[SSVEPExperiment] Waiting for IDUN recorder to be ready…")
        while True:
            elapsed = time.time() - start_t
            status_msg = getattr(self._recorder, "_status_msg", "starting...")
            lines = [
                f"Refresh rate: {refresh_rate:.1f} Hz",
                f"IDUN status: {status_msg}",
                f"Waiting for IDUN recorder: {int(elapsed)}s / {int(timeout)}s",
                "",
                "Press Esc or close the window to abort",
            ]
            try:
                self._update_status(lines)
                self._win.clearBuffer()
                if hasattr(self, "_status_box") and self._status_box is not None:
                    self._status_box.draw()
                if hasattr(self, "_status_stim") and self._status_stim is not None:
                    self._status_stim.draw()
                self._win.flip()
            except Exception:
                pass

            if self._abort_requested():
                print("[SSVEPExperiment] User requested abort during setup.", flush=True)
                raise ExperimentAborted("user aborted during setup")

            if self._recorder.is_ready:
                break
            if elapsed >= timeout:
                break
            time.sleep(poll)
        ready = self._recorder.is_ready
        if not ready:
            print(
                "[SSVEPExperiment] WARNING: IDUN recorder did not start within "
                "90 s. Continuing without EEG recording."
            )
        elif self._recorder.error:
            print(
                f"[SSVEPExperiment] WARNING: IDUN recorder error: "
                f"{self._recorder.error}. Continuing without EEG."
            )
        else:
            print(
                f"[SSVEPExperiment] IDUN recording live. "
                f"ID: {self._recorder.recording_id}"
            )

    def _measure_refresh_rate_interactive(self, min_seconds: float = 1.0, max_seconds: float = 3.0, target_samples: int = 90) -> float:
        """Measure refresh rate without blocking the UI.

        PsychoPy's getActualFrameRate() can sit on a blank screen while it
        looks for a stable estimate. This version flips frames in a loop,
        updates the status overlay, and keeps Escape / window-close responsive.
        """
        import statistics
        import time as _time

        samples: list[float] = []
        last_flip = None
        start = _time.perf_counter()
        warmup = 10

        while True:
            elapsed = _time.perf_counter() - start
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

            flip_time = _time.perf_counter()
            if last_flip is not None and warmup <= 0:
                samples.append(flip_time - last_flip)
            last_flip = flip_time
            warmup -= 1

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
        cfg = self._cfg
        layout = cfg.layout.lower()

        # blinking_stimuli image loading requires real files.
        # If the configured image_paths don't exist, generate placeholders.
        try:
            image_paths = self._resolve_image_paths(cfg.image_paths, len(cfg.flicker_frequencies))
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
                    image_paths = image_paths,
                    W=W, H=H,
                    n           = cfg.layout_circle_n,
                    radius      = cfg.layout_circle_radius,
                    size        = cfg.stimulus_size,
                    target_freq = cfg.flicker_frequencies,
                    flicker_mode = cfg.flicker_mode,
                    flicker_type = cfg.flicker_type,
                    color_mode   = cfg.color_mode,
                )
            elif layout == "checkerboard":
                print("[SSVEPExperiment] Creating checkerboard layout via QuickLayout.checkerboard", flush=True)

                rows = cfg.layout_rows
                cols = cfg.layout_cols
                n_cells = rows * cols

                # Always use resolved image paths
                if len(image_paths) < 2:
                    raise ValueError(
                        "Checkerboard layout requires at least 2 image paths (A and B)"
                    )

                img_a = image_paths[0]
                img_b = image_paths[1]

                freqs = cfg.flicker_frequencies

                # Case 1: exactly two frequencies (preferred)
                if isinstance(freqs, (list, tuple)) and len(freqs) == 2:
                    tf_a, tf_b = freqs

                # Case 2: full grid frequencies
                elif isinstance(freqs, (list, tuple)) and len(freqs) == n_cells:
                    tf_a = []
                    tf_b = []
                    for i in range(n_cells):
                        r = i // cols
                        c = i % cols
                        if (r + c) % 2 == 0:
                            tf_a.append(freqs[i])
                        else:
                            tf_b.append(freqs[i])

                # Case 3: single scalar frequency
                elif isinstance(freqs, (int, float)):
                    tf_a = tf_b = freqs

                else:
                    raise ValueError(
                        f"Invalid flicker_frequencies for checkerboard: {freqs}\n"
                        f"Expected: [A, B] or full list of {n_cells}"
                    )

                return QuickLayout.checkerboard(
                    image_paths_a = img_a,
                    image_paths_b = img_b,
                    W = W,
                    H = H,
                    rows = rows,
                    cols = cols,
                    size = cfg.stimulus_size,
                    target_freq_a = tf_a,
                    target_freq_b = tf_b,
                    flicker_mode = cfg.flicker_mode,
                    flicker_type = cfg.flicker_type,
                    color_mode = cfg.color_mode,
                )
            else:
                raise ValueError(
                    f"Unsupported layout '{cfg.layout}'. Choose 'grid', 'circle' or 'checkerboard'."
                )
        except BaseException as exc:
            import traceback
            print(f"[SSVEPExperiment] ERROR building stimuli ({layout}): {exc}", flush=True)
            traceback.print_exc()
            raise

    @staticmethod
    def _resolve_image_paths(paths: list, n_stimuli: int) -> list:
        """
        Return a list of valid image file paths of length n_stimuli.

        If a path doesn't exist, generate a coloured placeholder PNG using
        blinking_stimuli's _make_placeholder helper (requires pygame.font.init).
        """
        import pygame
        pygame.font.init()

        # Delayed import to avoid circular dependency issues
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

    # ── main trial loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full trial sequence."""
        cfg   = self._cfg
        win   = self._win
        clock = self._clock

        # Reset the experiment clock at the moment we consider "t = 0"
        clock.reset()

        self._log_event(EventMarker(
            event_type="EXPERIMENT_START",
            psychopy_time_s=clock.getTime(),
        ))

        # ── pre-experiment rest ───────────────────────────────────────────────
        self._show_message("Preparing…")
        core.wait(cfg.pre_experiment_rest_s)
        self._show_message("")

        # ── trial loop ────────────────────────────────────────────────────────
        for trial_idx in range(cfg.n_trials):
            if self._check_quit():
                raise ExperimentAborted("user aborted during run")

            self._run_trial(trial_idx)
            self._run_isi(trial_idx)

        # ── end ───────────────────────────────────────────────────────────────
        self._log_event(EventMarker(
            event_type="EXPERIMENT_END",
            psychopy_time_s=clock.getTime(),
            notes=f"total_trials={cfg.n_trials}",
        ))

        self._show_message("Experiment complete.\nThank you.")
        core.wait(2.0)

        self._save_log()

    def _run_trial(self, trial_idx: int) -> None:
        """Present all stimuli for one trial epoch."""
        cfg   = self._cfg
        win   = self._win
        clock = self._clock

        # Reset blinking_stimuli frame counters so all stimuli start in phase
        self._driver.reset_frames()

        trial_onset = clock.getTime()
        self._log_event(EventMarker(
            event_type="TRIAL_START",
            psychopy_time_s=trial_onset,
            trial_number=trial_idx + 1,
            notes=f"all_{len(self._stimuli)}_stimuli",
        ))

        # Push marker to IDUN background thread for recording annotation
        if self._recorder:
            self._recorder.push_marker(EventMarker(
                event_type="TRIAL_START",
                psychopy_time_s=trial_onset,
                trial_number=trial_idx + 1,
                idun_recording_id=self._recorder.recording_id or "",
            ))

        # ── stimulus presentation loop ────────────────────────────────────────
        while clock.getTime() - trial_onset < cfg.trial_duration_s:
            if self._check_quit():
                raise ExperimentAborted("user aborted during trial")

            win.clearBuffer()
            self._driver.draw()
            win.flip()
            self._driver.update()

        trial_end = clock.getTime()
        self._log_event(EventMarker(
            event_type="TRIAL_END",
            psychopy_time_s=trial_end,
            trial_number=trial_idx + 1,
            notes=f"duration_s={trial_end - trial_onset:.4f}",
        ))

        print(
            f"  Trial {trial_idx + 1:>3}/{cfg.n_trials}  "
            f"t={trial_onset:.3f}s  "
            f"dur={trial_end - trial_onset:.3f}s"
        )

    def _run_isi(self, trial_idx: int) -> None:
        """Blank inter-stimulus interval."""
        cfg   = self._cfg
        win   = self._win
        clock = self._clock

        isi_onset = clock.getTime()
        self._log_event(EventMarker(
            event_type="ISI_START",
            psychopy_time_s=isi_onset,
            trial_number=trial_idx + 1,
        ))

        while clock.getTime() - isi_onset < cfg.isi_duration_s:
            if self._check_quit():
                raise ExperimentAborted("user aborted during ISI")
            win.clearBuffer()
            win.flip()

    # ── teardown ──────────────────────────────────────────────────────────────

    def teardown(self) -> None:
        """Stop the IDUN recorder and close the PsychoPy window."""
        if self._recorder:
            print("[SSVEPExperiment] Stopping IDUN recorder…")
            try:
                self._recorder.stop()
            except Exception as exc:
                print(f"[SSVEPExperiment] Recorder stop error: {exc}", flush=True)

        if self._win:
            try:
                self._win.close()
            except Exception as exc:
                print(f"[SSVEPExperiment] Window close error: {exc}", flush=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log_event(self, marker: EventMarker) -> None:
        """Append a marker to the in-memory event log."""
        if self._recorder and self._recorder.recording_id:
            marker.idun_recording_id = self._recorder.recording_id
        self._event_log.append(marker)

    def _save_log(self) -> None:
        """Write the event log to a CSV file."""
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn  = self._cfg.log_filename.replace("{datetime}", ts)
        out = Path(self._cfg.output_dir) / fn

        fields = list(asdict(self._event_log[0]).keys()) if self._event_log else []
        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for marker in self._event_log:
                    writer.writerow(asdict(marker))
            print(f"[SSVEPExperiment] Event log saved → {out}")
        except Exception as exc:
            print(f"[SSVEPExperiment] Failed to save log: {exc}")

    def _show_message(self, text: str) -> None:
        """Display a centred text message for one frame."""
        msg = visual.TextStim(
            self._win, text=text, color=(1, 1, 1), height=30, units="pix",
            wrapWidth=self._win.size[0] * 0.8
        )
        self._win.clearBuffer()
        msg.draw()
        self._win.flip()

    def _update_status(self, lines: list) -> None:
        """Update the top-left monospace status display with given lines."""
        try:
            if not hasattr(self, "_status_stim") or self._status_stim is None:
                return
            self._status_stim.text = "\n".join(lines)
        except Exception:
            # Do not raise from a status update; it's purely cosmetic.
            pass

    def _abort_requested(self) -> bool:
        """Return True if the user pressed Escape/Q or closed the window."""
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
        """Return True if the user requested to abort the experiment."""
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
# Post-session data utilities
# ──────────────────────────────────────────────────────────────────────────────

def download_recording(api_token: str, recording_id: str,
                       file_types: list = None) -> None:
    """
    Standalone helper — download EEG (and optionally IMU / impedance) data
    for a completed recording without running the full experiment.

    Usage
    ─────
      from ssvep_experiment import download_recording
      download_recording("idun_xxx", "recording-id-here")
    """
    if not IDUN_AVAILABLE:
        print("[download_recording] idun-guardian-sdk is not installed.")
        return

    if file_types is None:
        file_types = [FileTypes.EEG]

    client = GuardianClient(api_token=api_token)
    for ft in file_types:
        print(f"[download_recording] Downloading {ft.name}…")
        try:
            client.download_file(recording_id=recording_id, file_type=ft)
            print(f"[download_recording] {ft.name} saved.")
        except Exception as exc:
            print(f"[download_recording] {ft.name} failed: {exc}")


def list_recordings(api_token: str, limit: int = 10) -> list:
    """Return a list of the most recent completed recordings."""
    if not IDUN_AVAILABLE:
        print("[list_recordings] idun-guardian-sdk is not installed.")
        return []
    client = GuardianClient(api_token=api_token)
    recordings = client.get_recordings(status="COMPLETED", limit=limit)
    for i, r in enumerate(recordings):
        print(f"  {i}: {r}")
    return recordings


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIGURE YOUR EXPERIMENT HERE ───────────────────────────────────────
    config = ExperimentConfig(
        # ── Display ────────────────────────────────────────────────────────
        monitor_name     = "testMonitor",   # PsychoPy monitor profile
        screen_index     = 0,
        fullscreen       = False,
        target_fps       = 60,              # ← set to your monitor's refresh rate

        # ── Stimuli ────────────────────────────────────────────────────────
        # Replace "placeholder.png" with your actual image path.
        # A colour-block placeholder will be generated automatically if the
        # file does not exist, so the experiment will still run for testing.
        image_paths      = ["Images/WhiteSquare1.png", "Images/BlackSquare1.png"],
        stimulus_size    = (150, 150),
        flicker_frequencies = [8.0, 10.0],
        layout           = "checkerboard",          # "grid", "checkerboard" or "circle"
        layout_rows      = 2,
        layout_cols      = 4,

        # ── Timing ─────────────────────────────────────────────────────────
        n_trials         = 3,
        trial_duration_s = 4.0,
        isi_duration_s   = 1.0,

        # ── IDUN Guardian ──────────────────────────────────────────────────
        # Leave idun_api_token blank to use the IDUN_API_TOKEN env variable.
        # Leave idun_device_address blank to auto-search for the earbud.
        enable_idun              = False,
        idun_api_token           = "",
        idun_device_address      = "",
        idun_recording_duration_s = 3600,
        idun_stream_live_eeg     = True,
        idun_impedance_check     = True,
        idun_impedance_duration_s = 10,
        idun_download_after      = True,

        # ── Output ─────────────────────────────────────────────────────────
        output_dir       = ".",
        log_filename     = "ssvep_events_{datetime}.csv"
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
