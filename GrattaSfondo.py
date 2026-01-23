import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import threading
import numpy as np

# --- 1. IL FIX DEVE ESSERE ESEGUITO PRIMA DI TUTTO ---
# Questo crea il modulo mancante che Torchvision ha rimosso
try:
    import torchvision.transforms.functional as F
    import torchvision.transforms
    # Se il modulo vecchio non esiste, lo creiamo artificialmente
    try:
        from torchvision.transforms import functional_tensor
    except ImportError:
        import types
        dummy_ft = types.ModuleType("torchvision.transforms.functional_tensor")
        dummy_ft.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = dummy_ft
        # Iniettiamo anche dentro transforms
        torchvision.transforms.functional_tensor = dummy_ft
except ImportError:
    pass 
# -----------------------------------------------------

# --- 2. ORA POSSIAMO IMPORTARE L'AI ---
try:
    import torch
    import cv2
    from PIL import Image, ImageQt, ImageFilter
    from rembg import remove
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError as e:
    print(f"Errore importazione librerie: {e}")

# --- 3. INFINE L'INTERFACCIA GRAFICA ---
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox, QInputDialog,
    QProgressBar
)

TARGET_W = 5000
TARGET_H = 5000

def resource_path(rel_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(sys.argv[0])
    return os.path.join(base, rel_path)

def fit_to_canvas(img: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    iw, ih = img.size
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    mode = "RGBA" if img.mode == "RGBA" else "RGB"
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new(mode, (canvas_w, canvas_h), (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255))
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas.paste(img_resized, (off_x, off_y), img_resized if img_resized.mode == "RGBA" else None)
    return canvas

class UpscaleThread(QThread):
    progress = Signal(int)
    finished = Signal(Image.Image, int, str)

    def __init__(self, image, scale, model_path, device):
        super().__init__()
        self.image = image
        self.scale = scale
        self.model_path = model_path
        self.device = device

    def _tile_for_size(self, w, h):
        mp = (w * h) / 1_000_000
        if mp > 25: return 64
        if mp > 12: return 96
        return 128

    def run(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modello AI mancante in: {self.model_path}")

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)

            src = self.image.convert("RGB")
            img_bgr = np.array(src)[:, :, ::-1].astype(np.uint8)
            h, w, _ = img_bgr.shape

            tile = self._tile_for_size(w, h)
            
            upsampler = RealESRGANer(
                scale=min(self.scale, 4),
                model_path=self.model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=False, 
                device=self.device
            )

            self.progress.emit(10)
            steps = max(10, (h * w) // (512 * 512))
            for s in range(1, steps + 1):
                self.progress.emit(min(90, int(10 + s / steps * 70)))
                self.msleep(60)

            output_bgr, _ = upsampler.enhance(img_bgr, outscale=min(self.scale, 4))

            if self.scale == 6:
                pil_up = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
                new_w = int(pil_up.width * 1.5)
                new_h = int(pil_up.height * 1.5)
                pil_up = pil_up.resize((new_w, new_h), Image.LANCZOS)
                output_bgr = cv2.cvtColor(np.array(pil_up), cv2.COLOR_RGB2BGR)

            output = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
            output = output.filter(ImageFilter.DETAIL).filter(ImageFilter.SHARPEN)
            output = fit_to_canvas(output, TARGET_W, TARGET_H)

            self.progress.emit(100)
            self.finished.emit(output, self.scale, "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, self.scale, str(e))

class GrattaSfondo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GrattaSfondo — by Marco Rinaldi")
        self.resize(820, 520)
        self.setAcceptDrops(True)
        self.path = None
        self.result_image = None
        self.current_image = None
        self.thread = None
        self.last_ai = False

        self.label_input = QLabel("Trascina qui un'immagine o clicca su 'Apri immagine'")
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setStyleSheet("border: 2px dashed #aaa; padding: 20px;")

        self.label_output = QLabel("Anteprima risultato")
        self.label_output.setAlignment(Qt.AlignCenter)
        self.label_output.setStyleSheet("border: 1px solid #ccc; background-color: #fafafa;")

        self.btn_open = QPushButton("Apri immagine")
        self.btn_remove = QPushButton("Scontorna")
        self.btn_ai = QPushButton("Migliora con AI")
        self.btn_save = QPushButton("Salva")

        self.btn_remove.setEnabled(False)
        self.btn_ai.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "TIFF"])
        self.format_combo.setFixedWidth(100)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_open)
        button_layout.addWidget(self.btn_remove)
        button_layout.addWidget(self.btn_ai)
        button_layout.addWidget(QLabel("Formato:"))
        button_layout.addWidget(self.format_combo)
        button_layout.addWidget(self.btn_save)

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.label_input, 1)
        preview_layout.addWidget(self.label_output, 1)

        layout = QVBoxLayout()
        layout.addLayout(preview_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_remove.clicked.connect(self.remove_bg)
        self.btn_ai.clicked.connect(self.enhance_ai)
        self.btn_save.clicked.connect(self.save_image)

        threading.Thread(target=self._preload_model, daemon=True).start()

    def _preload_model(self):
        try:
            import importlib
            importlib.import_module("rembg")
        except Exception: pass

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                self.load_image(file_path)
                break

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona immagine", "", "Immagini (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path: self.load_image(file_path)

    def load_image(self, path):
        self.path = path
        self.current_image = Image.open(path)
        self.result_image = None
        self.last_ai = False
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        pixmap = QPixmap(path).scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_input.setPixmap(pixmap)
        self.label_output.clear()
        self.btn_remove.setEnabled(True)
        self.btn_ai.setEnabled(True)
        self.btn_save.setEnabled(False)

    def remove_bg(self):
        if not self.path: return
        try:
            from rembg import remove
            img = Image.open(self.path).convert("RGBA")
            w, h = img.size
            max_side = max(TARGET_W, TARGET_H)
            if max(w, h) < max_side:
                ratio = max_side / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            result = remove(img, alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_structure_size=5)

            np_img = np.array(result)
            alpha = np_img[:, :, 3]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha = cv2.erode(alpha, kernel, iterations=1)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
            np_img[:, :, 3] = alpha
            result = Image.fromarray(np_img, mode="RGBA").filter(ImageFilter.DETAIL).filter(ImageFilter.SHARPEN)
            result = fit_to_canvas(result, TARGET_W, TARGET_H)

            self.result_image = result
            self.show_preview(result)
            self.btn_save.setEnabled(True)
            self.last_ai = False
            QMessageBox.information(self, "Scontorno completato",
                f"Immagine scontornata ({result.size[0]}×{result.size[1]} px)")

        except Exception as e:
            QMessageBox.warning(self, "Errore", f"Errore durante lo scontorno:\n{e}")

    def enhance_ai(self):
        base_img = self.result_image if self.result_image else self.current_image
        if not base_img:
            QMessageBox.warning(self, "Errore", "Apri prima un'immagine.")
            return

        factor, ok = QInputDialog.getItem(
            self, "Migliora con AI",
            "Seleziona fattore di ingrandimento:",
            ["2x", "4x", "6x"], 1, False
        )
        if not ok: return

        if factor == "6x":
            reply = QMessageBox.warning(self, "Attenzione", "Elaborazione pesante. Procedere?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No: return

        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"

        scale = int(factor.replace("x", ""))
        model_path = resource_path(os.path.join("models", "RealESRGAN_x4plus.pth"))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.thread = UpscaleThread(base_img.copy(), scale, model_path, device)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_ai_finished)
        self.thread.start()

    def on_ai_finished(self, output, scale, error_msg: str):
        if output is None:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Errore AI", "Errore durante l'elaborazione AI:\n\n" + (error_msg or "(nessun dettaglio)"))
            return

        self.result_image = output
        self.last_ai = True
        self.show_preview(output)
        self.btn_save.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        QMessageBox.information(self, "AI completato",
            f"Upscale x{scale} completato!\nRisoluzione finale: {output.size[0]}×{output.size[1]} px")

    def show_preview(self, img):
        qimage = ImageQt.ImageQt(img)
        pixmap = QPixmap.fromImage(qimage).scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_output.setPixmap(pixmap)

    def save_image(self):
        if not self.result_image: return
        formato = self.format_combo.currentText().lower()
        ext = "png" if formato == "png" else "tiff"
        base_name = os.path.splitext(os.path.basename(self.path))[0] if self.path else "immagine"
        suffix = "_AI" if self.last_ai else "_scontornata"
        default_name = f"{base_name}{suffix}.{ext}"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Salva immagine",
            os.path.join(os.path.dirname(self.path or ""), default_name),
            f"{formato.upper()} Files (*.{ext})"
        )

        if save_path:
            try:
                self.result_image.save(save_path, formato.upper(), quality=100)
                QMessageBox.information(self, "Completato", f"Immagine salvata:\n{save_path}")
            except Exception as e:
                QMessageBox.warning(self, "Errore", f"Errore nel salvataggio:\n{e}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, "Elaborazione in corso", "Uscire?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.thread.requestInterruption()
            self.thread.quit()
            self.thread.wait(3000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GrattaSfondo()
    window.show()
    sys.exit(app.exec())