"""Local IDUN BLE streamer with optional LabStreamingLayer output.

This module connects to an IDUN earbud over BLE using Bleak, starts the
measurement stream with the device command bytes discovered in
`guardian_ble.py`, and publishes the raw BLE notification payloads to LSL.

Because the packet structure is not decoded here, the default LSL stream
sends each packet as a hex string sample. That makes it immediately usable for
connectivity tests, timing checks, and PsychoPy synchronization. If you later
reverse the packet format, you can plug in a parser that converts packets to a
fixed-size numeric sample.

Typical use:

    from idun_lsl_streamer import IdunLSLStreamer, create_marker_outlet

    streamer = IdunLSLStreamer(target_name="IGE4T15")
    streamer.start()

    markers = create_marker_outlet()
    markers.push("experiment_start")
    ...
    markers.push("stimulus_on")
    ...
    streamer.stop()

The LSL side uses pylsl's standard StreamInfo/StreamOutlet API, and the stream
can be read from PsychoPy or any other LSL consumer.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner
import traceback

try:
    from pylsl import StreamInfo, StreamOutlet
except Exception as exc:  # pragma: no cover - handled at runtime
    StreamInfo = None  # type: ignore[assignment]
    StreamOutlet = None  # type: ignore[assignment]
    _PYLSL_IMPORT_ERROR = exc
else:
    _PYLSL_IMPORT_ERROR = None
print("idun_lsl_streamer: module loaded")


UUID_MEAS_EEGIMU = "beffd56c-c915-48f5-930d-4c1feee0fcc4"
UUID_MEAS_IMP = "beffd56c-c915-48f5-930d-4c1feee0fcc8"
UUID_CFG = "beffd56c-c915-48f5-930d-4c1feee0fcc9"
UUID_CMD = "beffd56c-c915-48f5-930d-4c1feee0fcca"

START_CMD = b"M"
STOP_CMD = b"S"
START_IMP_CMD = b"Z"
STOP_IMP_CMD = b"X"


@dataclass(slots=True)
class Packet:
    timestamp_utc: str
    sender: int
    stream: str
    payload: bytes

    @property
    def hex(self) -> str:
        return self.payload.hex()


class MarkerOutlet:
    """Small helper for PsychoPy or any other task code that wants LSL markers."""

    def __init__(
        self,
        name: str = "PsychoPyMarkers",
        stream_type: str = "Markers",
        source_id: str = "psychopy_markers",
    ) -> None:
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(
                "pylsl is not available. Install pylsl and liblsl first."
            ) from _PYLSL_IMPORT_ERROR

        info = StreamInfo(name, stream_type, 1, 0, "string", source_id)
        self._outlet = StreamOutlet(info)
        print(f"MarkerOutlet: created name={name} stream_type={stream_type} source_id={source_id}")

    def push(self, marker: str) -> None:
        print(f"MarkerOutlet: pushing marker={marker}")
        self._outlet.push_sample([marker])


class IdunLSLStreamer:
    """Connect to an IDUN earbud over BLE and publish raw packets to LSL.

    Parameters
    ----------
    target_name:
        BLE name to search for, e.g. "IGE4T15".
    target_address:
        Specific BLE address. If provided, this is preferred over name search.
    lsl_name:
        Name of the LSL stream that will carry raw packet hex strings.
    lsl_type:
        Stream type for the raw packet stream.
    source_id:
        Stable source identifier for LSL consumers.
    impedance_to_lsl:
        If True, impedance notifications are also published to a second LSL stream.
    log_path:
        Optional path to a JSONL file that receives every packet, write command,
        and connection event.
    packet_handler:
        Optional callable called as `packet_handler(packet)` for every notification.
        Use this to decode packets later if you reverse engineer the format.
    """

    def __init__(
        self,
        target_name: str = "IGE4T15",
        target_address: Optional[str] = None,
        lsl_name: str = "IDUN_RawPackets",
        lsl_type: str = "Raw",
        source_id: str = "idun_raw_packets",
        impedance_to_lsl: bool = False,
        log_path: Optional[str | Path] = None,
        packet_handler: Optional[Callable[[Packet], None]] = None,
    ) -> None:
        self.target_name = target_name
        self.target_address = target_address
        self.impedance_to_lsl = impedance_to_lsl
        self.packet_handler = packet_handler

        self._lsl_name = lsl_name
        self._lsl_type = lsl_type
        self._source_id = source_id
        print(
            f"IdunLSLStreamer: init target_name={target_name} target_address={target_address} "
            f"lsl_name={lsl_name} impedance_to_lsl={impedance_to_lsl} log_path={log_path}"
        )

        self._log_path = Path(log_path) if log_path else None
        self._client: Optional[BleakClient] = None
        self._device = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None

        self._packet_outlet = self._make_packet_outlet()
        self._imp_outlet = self._make_impedance_outlet() if impedance_to_lsl else None

    def _make_packet_outlet(self):
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(
                "pylsl is not available. Install pylsl and liblsl first."
            ) from _PYLSL_IMPORT_ERROR
        info = StreamInfo(self._lsl_name, self._lsl_type, 1, 0, "string", self._source_id)
        outlet = StreamOutlet(info)
        print(f"_make_packet_outlet: created outlet name={self._lsl_name} type={self._lsl_type} source_id={self._source_id}")
        return outlet

    def _make_impedance_outlet(self):
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(
                "pylsl is not available. Install pylsl and liblsl first."
            ) from _PYLSL_IMPORT_ERROR
        info = StreamInfo("IDUN_Impedance", "Impedance", 1, 0, "string", "idun_impedance")
        outlet = StreamOutlet(info)
        print("_make_impedance_outlet: created impedance outlet")
        return outlet

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log(self, kind: str, **payload) -> None:
        if self._log_path is None:
            return
        row = {"ts_utc": self._utc_now(), "kind": kind, **payload}
        print(f"_log: writing row={row} to {self._log_path}")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _find_device(self):
        print(f"_find_device: searching for target_address={self.target_address} target_name={self.target_name}")
        if self.target_address:
            device = await BleakScanner.find_device_by_address(self.target_address, timeout=8.0)
            print(f"_find_device: find_device_by_address returned {device}")
            return device

        devices = await BleakScanner.discover(timeout=8.0)
        print(f"_find_device: discover found {len(devices)} devices")
        for d in devices:
            print(f"_find_device: device name={d.name} address={d.address}")
            if d.name == self.target_name:
                print(f"_find_device: matched target_name on device {d}")
                return d
        print("_find_device: no matching device found")
        return None

    def _notify_packet(self, stream: str, sender: int, data: bytes) -> None:
        packet = Packet(
            timestamp_utc=self._utc_now(),
            sender=sender,
            stream=stream,
            payload=bytes(data),
        )
        print(f"_notify_packet: stream={stream} sender={sender} len={len(data)} hex={packet.hex}")
        self._log("packet", stream=stream, sender=sender, hex=packet.hex)
        if stream == "impedance" and self._imp_outlet is not None:
            print("_notify_packet: pushing to impedance outlet")
            self._imp_outlet.push_sample([packet.hex])
        else:
            print("_notify_packet: pushing to packet outlet")
            self._packet_outlet.push_sample([packet.hex])
        if self.packet_handler:
            print("_notify_packet: calling packet_handler")
            self.packet_handler(packet)

    def _make_callback(self, stream: str):
        print(f"_make_callback: creating callback for stream={stream}")

        def cb(sender: int, data: bytearray):
            print(f"callback: invoked for stream={stream} sender={sender} len={len(data)}")
            self._notify_packet(stream, sender, bytes(data))

        return cb

    async def _run_async(self) -> None:
        self._device = await self._find_device()
        if not self._device:
            raise RuntimeError(
                f"IDUN device not found. Looked for address={self.target_address!r} name={self.target_name!r}."
            )

        self._log("device_found", name=self._device.name, address=self._device.address)
        print(f"_run_async: device found name={self._device.name} address={self._device.address}")

        async with BleakClient(self._device, pair=True) as client:
            self._client = client
            self._log("connected", connected=bool(client.is_connected), address=self._device.address)
            print(f"_run_async: connected={client.is_connected} to {self._device.address}")

            await client.start_notify(UUID_MEAS_EEGIMU, self._make_callback("eegimu"))
            self._log("subscribed", uuid=UUID_MEAS_EEGIMU)
            print(f"_run_async: started notify for UUID_MEAS_EEGIMU")

            if self.impedance_to_lsl:
                await client.start_notify(UUID_MEAS_IMP, self._make_callback("impedance"))
                self._log("subscribed", uuid=UUID_MEAS_IMP)
                print(f"_run_async: started notify for UUID_MEAS_IMP")

            await client.write_gatt_char(UUID_CMD, START_CMD, response=False)
            self._log("write", uuid=UUID_CMD, value=START_CMD.decode("ascii"))
            print(f"_run_async: wrote START_CMD {START_CMD!r} to UUID_CMD")

            if self._stop_event is None:
                raise RuntimeError("Stop event not initialized")
            print("_run_async: waiting for stop event")
            await self._stop_event.wait()
            print("_run_async: stop event received, attempting to stop")

            try:
                await client.write_gatt_char(UUID_CMD, STOP_CMD, response=False)
                self._log("write", uuid=UUID_CMD, value=STOP_CMD.decode("ascii"))
                print(f"_run_async: wrote STOP_CMD {STOP_CMD!r} to UUID_CMD")
            finally:
                try:
                    await client.stop_notify(UUID_MEAS_EEGIMU)
                except Exception:
                    pass
                if self.impedance_to_lsl:
                    try:
                        await client.stop_notify(UUID_MEAS_IMP)
                    except Exception:
                        pass
                print("_run_async: cleaned up notifications and exiting")

    def _thread_main(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._stop_event = asyncio.Event()
            self._ready.set()
            self._loop.run_until_complete(self._run_async())
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            print("THREAD ERROR:", exc)
            traceback.print_exc()
        finally:
            try:
                if self._loop is not None:
                    self._loop.close()
            finally:
                self._loop = None
                self._client = None
                self._running = False

    def start(self) -> None:
        """Start the BLE connection and begin publishing packets to LSL."""
        if self._running:
            return
        self._running = True
        self._error = None
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=15.0)
        print(f"start: thread started, running={self._running}, error={self._error}")
        if self._error:
            raise self._error

    def stop(self) -> None:
        """Stop streaming and disconnect."""
        if not self._running:
            return
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        print("stop: stop_event set, waiting for thread to join")
        if self._thread is not None:
            self._thread.join(timeout=20.0)
        self._running = False
        print("stop: thread joined, running set to False")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._error


# Convenience functions for direct scripting use.

def create_marker_outlet(
    name: str = "PsychoPyMarkers",
    stream_type: str = "Markers",
    source_id: str = "psychopy_markers",
) -> MarkerOutlet:
    return MarkerOutlet(name=name, stream_type=stream_type, source_id=source_id)


def run_demo():
    streamer = IdunLSLStreamer(target_name="IGE4T15", impedance_to_lsl=True)

    try:
        streamer.start()

        try:
            print("Streaming... press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            streamer.stop()

    except Exception as e:
        print("ERROR:", e)

    finally:
        streamer.stop()
        print("Stopped")

        if streamer.last_error:
            print("THREAD ERROR:", streamer.last_error)


if __name__ == "__main__":
    run_demo()
