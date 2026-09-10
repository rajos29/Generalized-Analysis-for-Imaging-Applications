from __future__ import annotations

import re
import time
from queue import Empty, Queue

try:
    import serial
except ImportError:
    serial = None
from qtpy.QtCore import QThread, Signal

class Ender3Worker(QThread):
    status = Signal(str)
    connected_changed = Signal(bool)
    position_changed = Signal(float, float, float)
    log_line = Signal(str)

    def __init__(self, port: str, baud: int, max_x: float, max_y: float, max_z: float) -> None:
        super().__init__()
        self.port = port
        self.baud = baud
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.commands: Queue[tuple[str, object]] = Queue()
        self.running = True
        self.serial_port = None
        self.position: dict[str, float | None] = {"X": None, "Y": None, "Z": None}

    def run(self) -> None:
        if serial is None:
            self.status.emit("pyserial is not installed. Run: pip install pyserial")
            self.connected_changed.emit(False)
            return
        try:
            self.serial_port = serial.Serial(self.port, self.baud, timeout=0.15, write_timeout=1.0)
            time.sleep(2.0)
            self._drain_startup_lines()
            self.connected_changed.emit(True)
            self.status.emit(f"Connected to Ender 3 on {self.port} @ {self.baud}")
            self.enqueue_position_request()
            last_poll = time.monotonic()
            while self.running:
                try:
                    command, payload = self.commands.get(timeout=0.05)
                except Empty:
                    command = ""
                    payload = None
                if command:
                    self._handle_command(command, payload)
                if time.monotonic() - last_poll > 2.0:
                    self._send_gcode("M114")
                    last_poll = time.monotonic()
        except Exception as exc:
            self.status.emit(f"Ender connection failed: {exc}")
        finally:
            if self.serial_port is not None:
                try:
                    self.serial_port.close()
                except Exception:
                    pass
            self.connected_changed.emit(False)
            self.status.emit("Ender disconnected")

    def stop(self) -> None:
        self.running = False
        self.wait(1500)

    def enqueue_jog(self, axis: str, delta: float, feed_rate: int) -> None:
        self.commands.put(("jog", {"axis": axis, "delta": delta, "feed_rate": feed_rate}))

    def enqueue_home(self) -> None:
        self.commands.put(("home", None))

    def enqueue_position_request(self) -> None:
        self.commands.put(("position", None))

    def enqueue_soft_stop(self) -> None:
        self.commands.put(("soft_stop", None))

    def enqueue_emergency_stop(self) -> None:
        while not self.commands.empty():
            try:
                self.commands.get_nowait()
            except Empty:
                break
        self.commands.put(("emergency_stop", None))

    def _handle_command(self, command: str, payload: object) -> None:
        if command == "jog" and isinstance(payload, dict):
            axis = str(payload["axis"]).upper()
            delta = float(payload["delta"])
            feed_rate = int(payload["feed_rate"])
            if not self._jog_within_limits(axis, delta):
                self.status.emit(f"Blocked {axis}{delta:+0.3f}: outside configured soft limits")
                return
            self._send_gcode("G91")
            self._send_gcode(f"G1 {axis}{delta:0.3f} F{feed_rate}")
            self._send_gcode("G90")
            self._send_gcode("M114")
        elif command == "home":
            self._send_gcode("G28")
            self._send_gcode("M114")
        elif command == "position":
            self._send_gcode("M114")
        elif command == "soft_stop":
            self._send_gcode("M410")
            self._send_gcode("M114")
        elif command == "emergency_stop":
            self._send_gcode("M112")
            self.status.emit("Emergency stop sent. Printer firmware may require reset.")

    def _jog_within_limits(self, axis: str, delta: float) -> bool:
        current = self.position.get(axis)
        if current is None:
            return True
        maximum = {"X": self.max_x, "Y": self.max_y, "Z": self.max_z}.get(axis, 0.0)
        target = float(current) + delta
        return 0.0 <= target <= maximum

    def _drain_startup_lines(self) -> None:
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline and self.serial_port is not None:
            line = self.serial_port.readline().decode(errors="replace").strip()
            if line:
                self.log_line.emit(line)

    def _send_gcode(self, command: str) -> list[str]:
        if self.serial_port is None:
            return []
        self.log_line.emit(f"> {command}")
        self.serial_port.write(f"{command}\n".encode("ascii"))
        self.serial_port.flush()
        lines: list[str] = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = self.serial_port.readline().decode(errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            self.log_line.emit(line)
            self._parse_position(line)
            if line.lower().startswith("ok"):
                break
        return lines

    def _parse_position(self, line: str) -> None:
        if "X:" not in line or "Y:" not in line or "Z:" not in line:
            return
        values = dict((axis, float(value)) for axis, value in re.findall(r"([XYZ]):\s*(-?\d+(?:\.\d+)?)", line))
        if {"X", "Y", "Z"}.issubset(values):
            self.position.update(values)
            self.position_changed.emit(values["X"], values["Y"], values["Z"])


