import os
import sys
import subprocess

def install_torch_clean():
    print("🔄 1. Reinstallo Torch per recuperare i file originali...")
    # Forza la reinstallazione per sovrascrivere il file vuoto
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "torch==2.2.2", "torchvision==0.17.2"])

def apply_smart_fix():
    print("🛠️ 2. Applico la patch intelligente...")
    import torch
    # Trova il file problematico
    target_file = os.path.join(os.path.dirname(torch.__file__), "_numpy", "_ufuncs.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Errore: Non trovo {target_file}")
        return

    # Legge il contenuto originale
    with open(target_file, "r") as f:
        content = f.read()

    # Se il fix c'è già, non fa nulla
    if "name = 'dummy'" in content:
        print("✅ Il file è già stato patchato correttamente.")
        return

    # AGGIUNGE LA VARIABILE MANCANTE IN CIMA
    # Questo risolve il 'NameError' mantenendo tutte le funzioni come 'conjugate'
    patch = "name = 'dummy' # FIX PER PYINSTALLER\n"
    new_content = patch + content

    with open(target_file, "w") as f:
        f.write(new_content)
    
    print(f"✅ SUCCESS! Ho riparato {target_file}")
    print("   (Ho aggiunto la definizione mancante senza cancellare le funzioni)")

if __name__ == "__main__":
    try:
        install_torch_clean()
        apply_smart_fix()
        print("\n🎉 TUTTO PRONTO. Ora puoi lanciare PyInstaller.")
    except Exception as e:
        print(f"\n❌ Qualcosa è andato storto: {e}")