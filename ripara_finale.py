import os
import sys
import subprocess

def install_torch_clean():
    print("🔄 1. Pulisco e reinstallo Torch (attendere)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "torch==2.2.2", "torchvision==0.17.2"])
    except Exception as e:
        print(f"Errore installazione: {e}")
        sys.exit(1)

def apply_safe_fix():
    print("🛠️ 2. Applico la patch nel punto GIUSTO...")
    import torch
    target_file = os.path.join(os.path.dirname(torch.__file__), "_numpy", "_ufuncs.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Errore: Non trovo {target_file}")
        return

    with open(target_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    patched = False
    
    # Cerca la riga __future__ e inserisce il fix SUBITO DOPO
    for line in lines:
        new_lines.append(line)
        if "from __future__" in line and not patched:
            new_lines.append("\nname = 'dummy' # FIX SICURO PER PYINSTALLER\n")
            patched = True
    
    # Se per caso non c'era __future__, lo mette all'inizio
    if not patched:
        new_lines.insert(0, "name = 'dummy' # FIX SICURO PER PYINSTALLER\n")

    with open(target_file, "w") as f:
        f.writelines(new_lines)
    
    print(f"✅ SUCCESS! File riparato rispettando la sintassi Python.")

if __name__ == "__main__":
    install_torch_clean()
    apply_safe_fix()
    print("\n🎉 ORA PUOI LANCIARE PYINSTALLER.")