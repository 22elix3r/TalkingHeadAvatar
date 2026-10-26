"""Protocol fault fixture only. No model, synthesis or performance qualification."""

import struct
import sys
import threading
import time

HEADER = struct.Struct("<cIIQQ")
lock = threading.Lock()
cancelled = threading.Event()
mode = sys.argv[1]


def frame(kind, serial, count=0, payload=b""):
    with lock:
        sys.stdout.buffer.write(HEADER.pack(kind, 24000 if kind != b"E" else 0, count, 1000000, serial)+payload)
        sys.stdout.buffer.flush()


def generate(payload):
    serial = struct.unpack("<Q", payload[:8])[0]
    text = payload[8:].decode()
    frame(b"A", serial-1, 1, struct.pack("<f", 0.9))  # Stale request must be discarded.
    if mode == "bad-rate":
        with lock:
            sys.stdout.buffer.write(HEADER.pack(b"A", 16000, 1, 0, serial)+struct.pack("<f", 0.2))
            sys.stdout.buffer.flush()
        return
    count = 48000 if mode == "backpressure" else 100
    frame(b"A", serial, count, struct.pack(f"<{count}f", *([0.2]*count)))
    if mode == "failure":
        frame(b"E", serial, 15, b"injected error!")
        return
    if mode == "hang":
        while True:
            time.sleep(0.1)
    time.sleep(0.04)
    if cancelled.is_set():
        frame(b"C", serial)
        return
    frame(b"A", serial, 200, struct.pack("<200f", *([0.3]*200)))
    frame(b"A", serial, 17, struct.pack("<17f", *([0.1]*17)))
    frame(b"D", serial)
    frame(b"A", serial, 1, struct.pack("<f", 0.8))  # Post-EOF data cannot enter the stream.


if __name__ == "__main__":
    frame(b"R", 0)
    while True:
        kind = sys.stdin.buffer.read(1)
        if not kind:
            break
        size = struct.unpack("<I", sys.stdin.buffer.read(4))[0]
        payload = sys.stdin.buffer.read(size)
        if kind == b"G":
            cancelled.clear()
            threading.Thread(target=generate, args=(payload,), daemon=True).start()
        elif kind == b"C":
            cancelled.set()
        elif kind == b"X":
            break
