"""
IDUN Guardian – Device Stats + Live EEG Viewer
================================================
Requires:
    pip install idun-guardian-sdk matplotlib

Usage:
    python idun_viewer.py

Set your API key either as an environment variable:
    export IDUN_API_TOKEN=idun_xxxxxxxxxxxx

Or paste it directly into API_TOKEN below.
"""

import asyncio
import collections
import os
import threading

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from idun_guardian_sdk import GuardianClient

# ─────────────────────────────────────────────
#  CONFIG  –  fill in or set env var
# ─────────────────────────────────────────────
API_TOKEN     = os.environ.get("IDUN_API_TOKEN", "idun_eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkNjM2YmRlZC1hMzc1LTRhZDUtYjM0NC1jYjBkNzllZmVjYmEiLCJ1aWQiOiIxZjYzZTFjNy03YjAwLTQ3ODUtODUwYS1jMDc0OTI3YzE4OGUiLCJkaWQiOiIwMC0wMi01Qi0wMC1GRi0wRiIsImlhdCI6MTc3NTc0MDgwMy45MDU4OTR9._p2y7JgEOG8AMKBTimomYm9HrLJzjlVup2rHMWIhylc")
DEVICE_ADDRESS = ""          # leave blank to auto-search for IGEB
RECORDING_TIMER = 300        # seconds (5 min default; Ctrl+C to stop early)
MAINS_60HZ    = False        # True for US/Canada 60 Hz mains
WINDOW_SEC    = 10           # seconds of history shown in the live graph
EEG_SAMPLE_RATE = 250        # Hz – Guardian sample rate


# ─────────────────────────────────────────────
#  Shared live-data buffers  (thread-safe deques)
# ─────────────────────────────────────────────
MAX_SAMPLES  = WINDOW_SEC * EEG_SAMPLE_RATE
raw_eeg_buf_ch1  = collections.deque(maxlen=MAX_SAMPLES)
raw_eeg_buf_ch2  = collections.deque(maxlen=MAX_SAMPLES)

filt_eeg_buf_ch1 = collections.deque(maxlen=MAX_SAMPLES)
filt_eeg_buf_ch2 = collections.deque(maxlen=MAX_SAMPLES)

live_stats = {
    "battery":    "–",
    "impedance":  "–",
    "quality":    "–",
    "jaw_clench": 0,
    "heog":       "–",
    "status":     "Connecting…",
}

# Threading events for coordination
_recording_started = threading.Event()   # set when asyncio is ready → open graph
_gui_closed        = threading.Event()   # set when graph window is closed → stop recording


# ─────────────────────────────────────────────
#  Callbacks  (invoked from asyncio thread)
# ─────────────────────────────────────────────
def on_live_insights(event):
    msg = event.message

    for s in msg.get("raw_eeg", []):
        raw_eeg_buf_ch1.append(s.get("ch1", 0))
        raw_eeg_buf_ch2.append(s.get("ch2", 0))

    for s in msg.get("filtered_eeg", []):
        filt_eeg_buf_ch1.append(s.get("ch1", 0))
        filt_eeg_buf_ch2.append(s.get("ch2", 0))


def on_predictions(event):
    msg    = event.message
    ptype  = msg.get("predictionType", "")
    result = msg.get("result", {})
    if ptype == "QUALITY_SCORE":
        qs = result.get("quality_score", "–")
        live_stats["quality"] = f"{qs:.2f}" if isinstance(qs, float) else str(qs)
    elif ptype == "JAW_CLENCH":
        live_stats["jaw_clench"] += 1
    elif ptype == "BIN_HEOG":
        live_stats["heog"] = "◀ LEFT" if result.get("heog", 0) == -1 else "RIGHT ▶"


# ─────────────────────────────────────────────
#  Pre-recording diagnostics
# ─────────────────────────────────────────────
async def run_checks(client: GuardianClient):
    print("\n" + "═" * 52)
    print("  IDUN Guardian — Device Diagnostics")
    print("═" * 52)

    try:
        user = await asyncio.to_thread(client.get_user_info)
        print(f"  Account   : {user}")
    except Exception as e:
        print(f"  Account   : (unavailable – {e})")

    try:
        batt = await client.check_battery()
        live_stats["battery"] = f"{batt}%"
        bar = "█" * (batt // 10) + "░" * (10 - batt // 10)
        print(f"  Battery   : {batt}%  [{bar}]")
    except Exception as e:
        print(f"  Battery   : (unavailable – {e})")

    try:
        mac = await client.get_device_mac_address()
        print(f"  MAC Addr  : {mac}")
    except Exception as e:
        print(f"  MAC Addr  : (unavailable – {e})")

    # Impedance – 5-second snapshot
    print("\n  Measuring impedance for 5 s …")
    imp_readings: list = []

    def capture_impedance(data):
        imp_readings.append(data)
        live_stats["impedance"] = f"{data} Ω"

    async def impedance_task():
        await client.stream_impedance(
            handler=capture_impedance,
            mains_freq_60hz=MAINS_60HZ,
        )

    task = asyncio.create_task(impedance_task())
    await asyncio.sleep(5)
    client.stop_impedance()
    try:
        await task
    except Exception:
        pass

    if imp_readings:
        avg = sum(float(str(v).replace(",", "").replace(" ", ""))
                  for v in imp_readings if str(v).replace(",", "").replace(".", "").replace(" ", "").isdigit()
                  ) / max(len(imp_readings), 1)
        print(f"  Impedance Readings : {imp_readings}")
        print(f"  Impedance : {avg} Ω")
        status = "✅ Good (<300 kΩ)" if (avg < 300000) else "⚠ Check fit"
        print(f"  Status    : {status}")
    else:
        print("  Impedance : (no readings captured)")

    print("═" * 52 + "\n")


# ─────────────────────────────────────────────
#  Asyncio worker  (runs in background thread)
# ─────────────────────────────────────────────
async def async_main():
    live_stats["status"] = "Connecting…"
    print("\n  Connecting to IDUN Guardian …")

    kwargs = {"api_token": API_TOKEN}
    if DEVICE_ADDRESS:
        kwargs["address"] = DEVICE_ADDRESS

    client = GuardianClient(**kwargs)
    await client.connect_device()
    print("  ✅ Connected!\n")

    await run_checks(client)

    client.subscribe_live_insights(
        raw_eeg=True, filtered_eeg=True, imu=False,
        handler=on_live_insights,
    )
    client.subscribe_realtime_predictions(
        fft=False, jaw_clench=True, bin_heog=True, quality_score=True,
        handler=on_predictions,
    )

    live_stats["status"] = "● Recording"
    _recording_started.set()   # ← unblocks the main thread to open the graph

    print(f"  📡 Recording for {RECORDING_TIMER} s  (close the graph window to stop early)\n")

    try:
        record_task = asyncio.create_task(
            client.start_recording(recording_timer=RECORDING_TIMER)
        )
        # Poll for GUI close alongside the SDK recording coroutine
        while not record_task.done():
            if _gui_closed.is_set():
                record_task.cancel()
                break
            await asyncio.sleep(0.2)
        await record_task
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n  ⏹  Recording stopped.")

    live_stats["status"] = "● Disconnecting…"
    await client.disconnect_device()
    print("  Disconnected. Goodbye!\n")


def run_asyncio_in_thread():
    asyncio.run(async_main())


# ─────────────────────────────────────────────
#  Matplotlib live viewer  (MUST run on main thread)
# ─────────────────────────────────────────────
def launch_live_viewer():
    print("  Waiting for device connection before opening graph…")
    _recording_started.wait()   # block until asyncio is ready

    fig = plt.figure(figsize=(13, 7), facecolor="#0f0f1a")
    fig.canvas.manager.set_window_title("IDUN Guardian – Live EEG Viewer")

    gs = GridSpec(2, 1, figure=fig, hspace=0.45)
    ax_raw  = fig.add_subplot(gs[0])
    ax_filt = fig.add_subplot(gs[1])

    for ax, title, color in [
        (ax_raw,  "Raw EEG (ch1)",       "#00e5ff"),
        (ax_filt, "Filtered EEG (ch1)",  "#76ff03"),
    ]:
        ax.set_facecolor("#12122a")
        ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Time (s)",        color="#aaaacc", fontsize=9)
        ax.set_ylabel("Amplitude (µV)",  color="#aaaacc", fontsize=9)
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, color="#1e1e40", linewidth=0.6)
        ax.set_xlim(0, WINDOW_SEC)
        ax.set_ylim(-100, 100)

    line_raw_ch1,  = ax_raw.plot([], [], color="#00e5ff", lw=0.8, label="ch1")
    line_raw_ch2,  = ax_raw.plot([], [], color="#ff4081", lw=0.8, label="ch2")

    line_filt_ch1, = ax_filt.plot([], [], color="#76ff03", lw=0.8, label="ch1")
    line_filt_ch2, = ax_filt.plot([], [], color="#ffd740", lw=0.8, label="ch2")

    ax_raw.legend(loc="upper right")
    ax_filt.legend(loc="upper right")

    def update(_frame):
        raw1  = list(raw_eeg_buf_ch1)
        raw2  = list(raw_eeg_buf_ch2)
        filt1 = list(filt_eeg_buf_ch1)
        filt2 = list(filt_eeg_buf_ch2)

        n = min(len(raw1), len(raw2), len(filt1), len(filt2))

        if n > 1:
            t_shifted = [(i - (n - 1)) / EEG_SAMPLE_RATE + WINDOW_SEC for i in range(n)]

            line_raw_ch1.set_data(t_shifted, raw1[-n:])
            line_raw_ch2.set_data(t_shifted, raw2[-n:])

            line_filt_ch1.set_data(t_shifted, filt1[-n:])
            line_filt_ch2.set_data(t_shifted, filt2[-n:])

            # Auto-scale using both channels
            for ax, data in [
                (ax_raw, raw1[-n:] + raw2[-n:]),
                (ax_filt, filt1[-n:] + filt2[-n:])
            ]:
                mn, mx = min(data), max(data)
                pad = max((mx - mn) * 0.1, 1.0)
                ax.set_ylim(mn - pad, mx + pad)

        fig.suptitle(
            f"🔋 Battery: {live_stats['battery']}   "
            f"⚡ Imp: {live_stats['impedance']}   "
            f"📊 Quality: {live_stats['quality']}   "
            f"👁 HEOG: {live_stats['heog']}   "
            f"😬 Jaw clenches: {live_stats['jaw_clench']}   "
            f"{live_stats['status']}",
            color="#e0e0ff", fontsize=10, y=0.98,
        )
        return line_raw_ch1, line_raw_ch2, line_filt_ch1, line_filt_ch2

    # Keep a reference to the animation object – Python GC will delete it otherwise
    anim = animation.FuncAnimation(  # noqa: F841
        fig, update,
        interval=50,            # ~20 fps
        blit=False,
        cache_frame_data=False,
    )

    def on_close(_event):
        print("\n  Graph window closed – stopping recording…")
        _gui_closed.set()

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()   # blocks here until the window is closed – correct on main thread


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Start asyncio (device + recording) in a background thread
    worker = threading.Thread(target=run_asyncio_in_thread, daemon=True)
    worker.start()

    # 2. Open the matplotlib GUI on the main thread (required by Qt/Tk on Windows)
    launch_live_viewer()

    # 3. Wait for the asyncio worker to finish cleanly
    worker.join(timeout=15)