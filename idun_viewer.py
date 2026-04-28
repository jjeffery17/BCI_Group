"""
IDUN Guardian - Device Stats + Live EEG Viewer
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
import time

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
#  Shared live-data buffers (thread-safe deques)
# ─────────────────────────────────────────────
MAX_SAMPLES  = WINDOW_SEC * EEG_SAMPLE_RATE
raw_eeg_buf  = collections.deque(maxlen=MAX_SAMPLES)
filt_eeg_buf = collections.deque(maxlen=MAX_SAMPLES)
ts_buf       = collections.deque(maxlen=MAX_SAMPLES)

# Running stats shown in the title bar
live_stats = {
    "battery":   "-",
    "impedance": "-",
    "quality":   "-",
    "jaw_clench": 0,
    "heog":      "-",
}


# ─────────────────────────────────────────────
#  EEG / insight callback
# ─────────────────────────────────────────────
def on_live_insights(event):
    msg = event.message
    raw  = msg.get("raw_eeg", [])
    filt = msg.get("filtered_eeg", [])
    for s in raw:
        raw_eeg_buf.append(s.get("ch1", 0))
        ts_buf.append(s.get("timestamp", time.time()))
    for s in filt:
        filt_eeg_buf.append(s.get("ch1", 0))


def on_predictions(event):
    msg  = event.message
    ptype = msg.get("predictionType", "")
    result = msg.get("result", {})
    if ptype == "QUALITY_SCORE":
        qs = result.get("quality_score", "–")
        live_stats["quality"] = f"{qs:.2f}" if isinstance(qs, float) else qs
    elif ptype == "JAW_CLENCH":
        live_stats["jaw_clench"] += 1
    elif ptype == "BIN_HEOG":
        h = result.get("heog", 0)
        live_stats["heog"] = "◀ LEFT" if h == -1 else "RIGHT ▶"


# ─────────────────────────────────────────────
#  Pre-recording checks (battery + impedance)
# ─────────────────────────────────────────────
async def run_checks(client: GuardianClient):
    print("\n" + "═" * 52)
    print("  IDUN Guardian — Device Diagnostics")
    print("═" * 52)

    # User info
    try:
        user = await asyncio.to_thread(client.get_user_info)
        print(f"  Account   : {user}")
    except Exception as e:
        print(f"  Account   : (unavailable – {e})")

    # Battery
    try:
        batt = await client.check_battery()
        live_stats["battery"] = f"{batt}%"
        bar = "█" * (batt // 10) + "░" * (10 - batt // 10)
        print(f"  Battery   : {batt}%  [{bar}]")
    except Exception as e:
        print(f"  Battery   : (unavailable – {e})")

    # MAC / device address
    try:
        mac = await client.get_device_mac_address()
        print(f"  MAC Addr  : {mac}")
    except Exception as e:
        print(f"  MAC Addr  : (unavailable – {e})")

    # Impedance – stream for 5 seconds then stop
    print("\n  Measuring impedance for 5 s …")
    imp_readings = []

    def capture_impedance(data):
        imp_readings.append(data)
        if imp_readings:
            live_stats["impedance"] = f"{data} Ω"

    async def impedance_task():
        await client.stream_impedance(handler=capture_impedance, mains_freq_60hz=MAINS_60HZ)

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
#  Matplotlib live-update graph
# ─────────────────────────────────────────────
def launch_live_viewer():
    """Runs the matplotlib window in the main thread."""
    fig = plt.figure(figsize=(13, 7), facecolor="#0f0f1a")
    fig.canvas.manager.set_window_title("IDUN Guardian – Live EEG Viewer")

    gs = GridSpec(2, 1, figure=fig, hspace=0.45)
    ax_raw  = fig.add_subplot(gs[0])
    ax_filt = fig.add_subplot(gs[1])

    for ax, title, color in [
        (ax_raw,  "Raw EEG (ch1)",      "#00e5ff"),
        (ax_filt, "Filtered EEG (ch1)", "#76ff03"),
    ]:
        ax.set_facecolor("#12122a")
        ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Time (s)",       color="#aaaacc", fontsize=9)
        ax.set_ylabel("Amplitude (µV)", color="#aaaacc", fontsize=9)
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, color="#1e1e40", linewidth=0.6)

    line_raw,  = ax_raw.plot([], [], color="#00e5ff", lw=0.8, alpha=0.9)
    line_filt, = ax_filt.plot([], [], color="#76ff03", lw=0.8, alpha=0.9)

    def update(_frame):
        # --- raw EEG ---
        raw  = list(raw_eeg_buf)
        filt = list(filt_eeg_buf)
        n    = min(len(raw), len(filt))

        if n > 1:
            t = [i / EEG_SAMPLE_RATE for i in range(n)]
            t_window = t[-MAX_SAMPLES:]
            raw_w  = raw[-MAX_SAMPLES:]
            filt_w = filt[-MAX_SAMPLES:]
            # re-zero time so x-axis shows "last N seconds"
            t_zero = [v - t_window[-1] + WINDOW_SEC for v in t_window]

            line_raw.set_data(t_zero, raw_w)
            line_filt.set_data(t_zero, filt_w)

            for ax, data in [(ax_raw, raw_w), (ax_filt, filt_w)]:
                if data:
                    mn, mx = min(data), max(data)
                    pad = max((mx - mn) * 0.1, 1)
                    ax.set_xlim(0, WINDOW_SEC)
                    ax.set_ylim(mn - pad, mx + pad)

        # --- suptitle stats ---
        fig.suptitle(
            f"🔋 Battery: {live_stats['battery']}   "
            f"⚡ Impedance: {live_stats['impedance']}   "
            f"📊 Quality: {live_stats['quality']}   "
            f"👁 HEOG: {live_stats['heog']}   "
            f"😬 Jaw clenches: {live_stats['jaw_clench']}",
            color="#e0e0ff", fontsize=10, y=0.98,
        )
        return line_raw, line_filt

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


# ─────────────────────────────────────────────
#  Main async flow
# ─────────────────────────────────────────────
async def main():
    print("\n  Connecting to IDUN Guardian …")

    kwargs = {"api_token": API_TOKEN}
    if DEVICE_ADDRESS:
        kwargs["address"] = DEVICE_ADDRESS

    client = GuardianClient(**kwargs)

    # 1. Connect
    await client.connect_device()
    print("  ✅ Connected!\n")

    # 2. Run diagnostics
    await run_checks(client)

    # 3. Subscribe to live insights + predictions
    client.subscribe_live_insights(raw_eeg=True, filtered_eeg=True, imu=False, handler=on_live_insights)
    client.subscribe_realtime_predictions(
        fft=False, jaw_clench=True, bin_heog=True, quality_score=True,
        handler=on_predictions,
    )

    # 4. Launch the matplotlib window in a daemon thread so asyncio keeps running
    viewer_thread = threading.Thread(target=launch_live_viewer, daemon=True)
    viewer_thread.start()

    print(f"  📡 Recording for {RECORDING_TIMER} s  (Ctrl+C to stop early)")
    print("  Live graph window should appear now.\n")

    # 5. Start recording (blocks until timer or Ctrl+C)
    try:
        await client.start_recording(recording_timer=RECORDING_TIMER)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n  ⏹  Recording interrupted by user.")

    # 6. Disconnect
    await client.disconnect_device()
    print("  Disconnected. Goodbye!\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())