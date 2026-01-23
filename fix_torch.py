import os
import torch
target = os.path.join(os.path.dirname(torch.__file__), "_numpy", "_ufuncs.py")
print(f"Riparo: {target}")
with open(target, "w") as f:
    f.write("pass")
print("✅ FATTO. File svuotato.")