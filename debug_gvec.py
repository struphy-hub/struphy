import faulthandler
faulthandler.enable()

import sys
import traceback

print("1. importing struphy", flush=True)

from struphy.geometry import domains

print("2. imported domains", flush=True)

try:
    print("3. constructing GVECunit()", flush=True)
    domain = domains.GVECunit()
    print("4. constructed successfully", flush=True)

except Exception:
    print("Python exception:")
    traceback.print_exc()
    raise

print("5. finished", flush=True)
