"""Decode IDUN raw BLE packets and visualize them through LabStreamingLayer.

This module is the next layer on top of `idun_lsl_streamer.py`.

Pipeline
--------
1. `IdunLSLStreamer` publishes raw BLE notifications as hex-string LSL samples.
2. `IdunLSLDecoder` consumes that raw LSL stream.
3. The decoder converts each hex payload into a numeric vector and publishes a
   decoded LSL stream.
4. `DecodedStreamPlotter` subscribes to the decoded stream and shows a live graph.

Important
---------
The exact IDUN packet structure is not documented here, so the decoder uses a
practical first-pass interpretation: each BLE notification payload is converted
into little-endian signed 16-bit integers.

That is enough to:
- verify the stream is live,
- stabilize the data path through LSL,
- inspect packet shape over time,
- and provide something PsychoPy / analysis tools can subscribe to.

If you later derive the true packet format, only `decode_idun_packet()` needs to
change.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np

try:
    from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop
except Exception as exc:  # pragma: no cover
    StreamInlet = None  # type: ignore[assignment]
    StreamInfo = None  # type: ignore[assignment]
    StreamOutlet = None  # type: ignore[assignment]
    resolve_byprop = None  # type: ignore[assignment]
    _PYLSL_IMPORT_ERROR = exc
else:
    _PYLSL_IMPORT_ERROR = None


RAW_STREAM_NAME = "IDUN_RawPackets"
RAW_STREAM_TYPE = "Raw"
DECODED_STREAM_NAME = "IDUN_Decoded"
DECODED_STREAM_TYPE = "EEG"
MARKER_STREAM_NAME = "IDUN_Markers"
MARKER_STREAM_TYPE = "Markers"


@dataclass(slots=True)
class DecodedPacket:
    timestamp: float
    raw_hex: str
    values: np.ndarray


def _require_pylsl() -> None:
    if StreamInlet is None or StreamInfo is None or StreamOutlet is None or resolve_byprop is None:
        raise RuntimeError("pylsl is not available. Install pylsl and liblsl first.") from _PYLSL_IMPORT_ERROR


def decode_idun_packet(payload: bytes) -> np.ndarray:
    """Decode one raw IDUN packet into a numeric vector.

    Current interpretation:
    - keep the payload as-is
    - if the length is odd, drop the final trailing byte
    - reinterpret the bytes as little-endian signed int16 values

    Returns
    -------
    np.ndarray
        1D float32 array suitable for LSL publishing or plotting.
    """
    if not payload:
        return np.empty(0, dtype=np.float32)

    usable_len = len(payload) - (len(payload) % 2)
    if usable_len == 0:
        return np.empty(0, dtype=np.float32)

    arr = np.frombuffer(payload[:usable_len], dtype="<i2").astype(np.float32)
    return arr


class MarkerOutlet:
    """Simple LSL marker outlet for PsychoPy events."""

    def __init__(self, name: str = MARKER_STREAM_NAME, stream_type: str = MARKER_STREAM_TYPE):
        _require_pylsl()
        info = StreamInfo(name, stream_type, 1, 0, "string", f"{name.lower()}_source")
        self._outlet = StreamOutlet(info)

    def push(self, marker: str) -> None:
        self._outlet.push_sample([marker])


class IdunLSLDecoder:
    """Read raw IDUN packets from LSL, decode them, and republish as numeric LSL.

    Parameters
    ----------
    raw_stream_name:
        The name of the raw packet LSL stream created by `IdunLSLStreamer`.
    decoded_stream_name:
        Name of the numeric decoded stream published by this class.
    max_queue:
        Internal queue size for decoded packets.
    """

    def __init__(
        self,
        raw_stream_name: str = RAW_STREAM_NAME,
        decoded_stream_name: str = DECODED_STREAM_NAME,
        decoded_stream_type: str = DECODED_STREAM_TYPE,
        max_queue: int = 500,
    ) -> None:
        _require_pylsl()
        self.raw_stream_name = raw_stream_name
        self.decoded_stream_name = decoded_stream_name
        self.decoded_stream_type = decoded_stream_type
        self._decoded_outlet: Optional[StreamOutlet] = None
        self._decoded_channels: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[DecodedPacket]" = queue.Queue(maxsize=max_queue)
        self._error: Optional[BaseException] = None
        self._inlet: Optional[StreamInlet] = None

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._error

    @property
    def decoded_channels(self) -> Optional[int]:
        return self._decoded_channels

    def _make_outlet(self, channels: int) -> StreamOutlet:
        info = StreamInfo(
            self.decoded_stream_name,
            self.decoded_stream_type,
            channels,
            0.0,
            "float32",
            f"{self.decoded_stream_name.lower()}_source",
        )
        return StreamOutlet(info)

    def _ensure_outlet(self, channels: int) -> None:
        if self._decoded_outlet is None:
            self._decoded_channels = channels
            self._decoded_outlet = self._make_outlet(channels)

    def _push_queue(self, packet: DecodedPacket) -> None:
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(packet)

    def _reader_loop(self) -> None:
        streams = resolve_byprop("name", self.raw_stream_name, timeout=10)
        if not streams:
            raise RuntimeError(f'Could not find raw LSL stream named "{self.raw_stream_name}"')

        self._inlet = StreamInlet(streams[0], max_buflen=5)

        while not self._stop.is_set():
            sample, timestamp = self._inlet.pull_sample(timeout=0.5)
            if sample is None:
                continue

            if not sample:
                continue

            raw_hex = str(sample[0])
            try:
                payload = bytes.fromhex(raw_hex)
            except ValueError:
                continue

            values = decode_idun_packet(payload)
            if values.size == 0:
                continue

            self._ensure_outlet(int(values.size))
            assert self._decoded_outlet is not None
            self._decoded_outlet.push_sample(values.tolist())

            packet = DecodedPacket(timestamp=timestamp, raw_hex=raw_hex, values=values)
            self._push_queue(packet)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._thread_run, daemon=True)
        self._thread.start()
        time.sleep(0.1)
        if self._error:
            raise self._error

    def _thread_run(self) -> None:
        try:
            self._reader_loop()
        except BaseException as exc:
            self._error = exc

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_packet_nowait(self) -> Optional[DecodedPacket]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class DecodedStreamPlotter:
    """Live plot of the decoded LSL stream using matplotlib.

    By default the first few channels are plotted because the decoded packets can
    be long. You can change `plot_channels` to display a different subset.
    """

    def __init__(
        self,
        stream_name: str = DECODED_STREAM_NAME,
        plot_channels: int = 8,
        window_samples: int = 250,
    ) -> None:
        _require_pylsl()
        self.stream_name = stream_name
        self.plot_channels = plot_channels
        self.window_samples = window_samples
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._error

    def start(self) -> None:
        # Run plotting in the main thread — matplotlib GUIs must run on main thread.
        self._stop.clear()
        self._error = None
        self._run()

    def stop(self) -> None:
        # Signal the plot to stop. There is no separate plot thread to join.
        self._stop.set()

    def _run(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        streams = resolve_byprop("name", self.stream_name, timeout=15)
        if not streams:
            raise RuntimeError(f'Could not find decoded LSL stream named "{self.stream_name}"')

        inlet = StreamInlet(streams[0], max_buflen=10)
        info = inlet.info()
        channel_count = int(info.channel_count())
        n_plot = min(self.plot_channels, channel_count)

        buffers: list[Deque[float]] = [deque(maxlen=self.window_samples) for _ in range(n_plot)]

        fig, ax = plt.subplots()
        lines = []
        for _ in range(n_plot):
            (line,) = ax.plot([])
            lines.append(line)

        ax.set_title(self.stream_name)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")

        def update(_frame):
            if self._stop.is_set():
                plt.close(fig)
                return lines

            while True:
                sample, _ts = inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                for i in range(n_plot):
                    buffers[i].append(float(sample[i]))

            if any(len(b) for b in buffers):
                all_vals = [v for b in buffers for v in b]
                ymin = min(all_vals)
                ymax = max(all_vals)
                if math.isclose(ymin, ymax):
                    ymin -= 1.0
                    ymax += 1.0
                ax.set_ylim(ymin, ymax)

            for i, line in enumerate(lines):
                y = list(buffers[i])
                x = list(range(len(y)))
                line.set_data(x, y)
                line.set_label(f"Ch {i+1}")

            ax.set_xlim(0, self.window_samples)
            ax.legend(loc="upper right")
            return lines

        # Keep a reference to the animation on the instance to avoid it being
        # garbage-collected and triggering the "Animation was deleted" warning.
        self._ani = FuncAnimation(fig, update, interval=20, blit=False)
        plt.show()


class IdunDecodeAndPlotApp:
    """Convenience wrapper that starts the decoder and then opens the plot."""

    def __init__(
        self,
        raw_stream_name: str = RAW_STREAM_NAME,
        decoded_stream_name: str = DECODED_STREAM_NAME,
        plot_channels: int = 8,
    ) -> None:
        self.decoder = IdunLSLDecoder(
            raw_stream_name=raw_stream_name,
            decoded_stream_name=decoded_stream_name,
        )
        self.plotter = DecodedStreamPlotter(
            stream_name=decoded_stream_name,
            plot_channels=plot_channels,
        )

    def run(self) -> None:
        self.decoder.start()
        try:
            # BLOCKING call — runs GUI in main thread
            self.plotter.start()
        finally:
            # Stop the decoder once the GUI loop exits.
            self.decoder.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode IDUN raw LSL packets and plot them.")
    parser.add_argument("--raw", default=RAW_STREAM_NAME, help="Raw LSL stream name")
    parser.add_argument("--decoded", default=DECODED_STREAM_NAME, help="Decoded LSL stream name")
    parser.add_argument("--channels", type=int, default=8, help="How many channels to plot")
    args = parser.parse_args()

    app = IdunDecodeAndPlotApp(
        raw_stream_name=args.raw,
        decoded_stream_name=args.decoded,
        plot_channels=args.channels,
    )
    app.run()


if __name__ == "__main__":
    main()
