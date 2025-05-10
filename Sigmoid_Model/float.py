# Python program to demonstrate adding two IEEE-754 double-precision numbers
import struct

def hex_to_double(h: str) -> float:
    """Convert IEEE-754 double in hex string (e.g. '0x3FF0000000000000') to Python float."""
    return struct.unpack('>d', struct.pack('>Q', int(h, 16)))[0]

def double_to_hex(f: float) -> str:
    """Convert a Python float to IEEE-754 double in hex string."""
    return hex(struct.unpack('>Q', struct.pack('>d', f))[0])

# Example values (1.0 + 2.0)
h1 = "0x0010000000000001"  # 1.0
h2 = "0x0010000000000000"  # 2.0

f1 = hex_to_double(h1)
f2 = hex_to_double(h2)
s = f1 - f2

print(f"sub {h1} ({f1}) - {h2} ({f2})")
print(f"Result float   : {s}")
print(f"Result hex     : {double_to_hex(s)}")
print(f"Result binary  : {bin(struct.unpack('>Q', struct.pack('>d', s))[0])}")
