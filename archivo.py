import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from pathlib import Path
import shutil


class SegmentAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentador de Imágenes con YOLO")

        # Cargar modelo
        try:
            self.model = YOLO('./gpuunl/best4.pt')
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
            exit()

        # UI Elements
        self.btn_select = tk.Button(root, text="Seleccionar Imagen", command=self.select_image)
        self.btn_select.pack(pady=10)

        self.label_original = tk.Label(root, text="Imagen Original")
        self.label_original.pack()
        self.panel_original = tk.Label(root)
        self.panel_original.pack(side="top")

        self.label_segmented = tk.Label(root, text="Imagen Segmentada")
        self.label_segmented.pack()
        self.panel_segmented = tk.Label(root)
        self.panel_segmented.pack(side="bottom")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            print(f"🖼️ Archivo seleccionado: {file_path}")
            self.full_process(file_path)

    def full_process(self, ruta_imagen):
        try:
            # Variables dinámicas según la imagen seleccionada
            single_image_path = Path(ruta_imagen)
            output_folder_name = "secciones_personalizadas2"
            output_dir = single_image_path.parent / output_folder_name
            tile_width = 257
            imagen = cv2.imread(ruta_imagen)
            tile_height = altura = imagen.shape[0]

            # 1. Dividir imagen en tiles
            self.process_single_image(single_image_path, output_dir, tile_width, tile_height)

            # 2. Configuración de directorios
            input_dir = str(output_dir)
            output_dir_visual = './Muestras/secciones_procesadas'
            output_dir_binary = './segmented_images1'
            os.makedirs(output_dir_visual, exist_ok=True)
            os.makedirs(output_dir_binary, exist_ok=True)

            # 3. Procesar todas las imágenes con YOLO
            imagenes = glob.glob(os.path.join(input_dir, '*.jpg'))
            if not imagenes:
                raise FileNotFoundError("❌ No se encontraron imágenes para procesar.")

            for img_path in imagenes:
                self.process_image(img_path, output_dir_visual, output_dir_binary)

            # 4. Reconstruir imagen completa a partir de máscaras
            output_dir_reconstructed = Path("./Resultado_Final")
            output_dir_reconstructed.mkdir(exist_ok=True)
            self.reconstruct_image_from_tiles(output_dir_binary, output_dir_reconstructed, single_image_path.stem)

            # 5. Eliminar carpetas temporales
            carpetas_temporales = [
                './Muestras/secciones_personalizadas2',
                './Muestras/secciones_procesadas',
                './segmented_images1'
            ]

            for carpeta in carpetas_temporales:
                if os.path.exists(carpeta):
                    try:
                        shutil.rmtree(carpeta)
                        print(f"🗑️ Carpeta eliminada: {carpeta}")
                    except Exception as e:
                        print(f"❌ Error al eliminar {carpeta}: {e}")


            # Mostrar resultados finales
            reconstructed_path = str(output_dir_reconstructed / f"{single_image_path.stem}_full_image_segmented.jpg")
            self.show_final_images(ruta_imagen, reconstructed_path)

        except Exception as e:
            messagebox.showerror("Error", f"Hubo un error durante el proceso: {e}")

    def process_single_image(self, image_path, output_dir, tile_width, tile_height):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        output_dir.mkdir(exist_ok=True)
        height, width, _ = image.shape
        num_tiles_x = width // tile_width
        num_tiles_y = height // tile_height

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                x_start = x * tile_width
                y_start = y * tile_height
                tile = image[y_start:y_start+tile_height, x_start:x_start+tile_width]
                tile_path = output_dir / f"section_{y}_{x}.jpg"
                cv2.imwrite(str(tile_path), tile)

        if height % tile_height != 0:
            y_start = num_tiles_y * tile_height
            for x in range(num_tiles_x):
                x_start = x * tile_width
                tile = image[y_start:, x_start:x_start+tile_width]
                tile_path = output_dir / f"section_{num_tiles_y}_{x}.jpg"
                cv2.imwrite(str(tile_path), tile)

        if width % tile_width != 0:
            x_start = num_tiles_x * tile_width
            for y in range(num_tiles_y):
                y_start = y * tile_height
                tile = image[y_start:y_start+tile_height, x_start:]
                tile_path = output_dir / f"section_{y}_{num_tiles_x}.jpg"
                cv2.imwrite(str(tile_path), tile)

        if width % tile_width != 0 and height % tile_height != 0:
            x_start = num_tiles_x * tile_width
            y_start = num_tiles_y * tile_height
            tile = image[y_start:, x_start:]
            tile_path = output_dir / f"section_{num_tiles_y}_{num_tiles_x}.jpg"
            cv2.imwrite(str(tile_path), tile)

    def process_image(self, image_path, output_dir_visual, output_dir_binary):
        image = cv2.imread(image_path)
        results = self.model(image_path)
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return

        # Máscara visual
        seg_visual = result.plot(boxes=False)
        out_vis = os.path.join(output_dir_visual, os.path.basename(image_path))
        cv2.imwrite(out_vis, seg_visual)

        # Máscara binaria
        segmented = np.zeros_like(image)
        for mask in result.masks.data:
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
            segmented[mask_resized > 0] = 255
        out_bin = os.path.join(output_dir_binary, os.path.basename(image_path))
        cv2.imwrite(out_bin, segmented)

    def reconstruct_image_from_tiles(self, tile_dir, output_dir, base_name):
        files = sorted(glob.glob(os.path.join(tile_dir, '*.jpg')))
        if not files:
            raise FileNotFoundError("No hay archivos para reconstruir.")
        sample = cv2.imread(files[0])
        tile_width = sample.shape[1]

        tiles_info = []
        for f in files:
            filename = Path(f).stem
            try:
                y = int(filename.split('_')[1])
                x = int(filename.split('_')[2])
            except IndexError:
                continue
            img = cv2.imread(f)
            h, w = img.shape[:2]
            tiles_info.append({'x': x, 'y': y, 'img': img, 'width': w, 'height': h})

        num_cols = max(t['x'] for t in tiles_info) + 1
        num_rows = max(t['y'] for t in tiles_info) + 1

        row_heights = [0]*num_rows
        col_widths = [0]*num_cols
        for t in tiles_info:
            row_heights[t['y']] = max(row_heights[t['y']], t['height'])
            col_widths[t['x']] = max(col_widths[t['x']], t['width'])

        total_width = sum(col_widths)
        total_height = sum(row_heights)
        full_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        for t in tiles_info:
            x_start = sum(col_widths[:t['x']])
            y_start = sum(row_heights[:t['y']])
            full_img[y_start:y_start+t['height'], x_start:x_start+t['width']] = t['img']

        # Renombrar el archivo final con el nombre base de la imagen original
        out_path = os.path.join(output_dir, f"{base_name}_full_image_segmented.jpg")
        cv2.imwrite(out_path, full_img)
        print(f"✅ Imagen completa reconstruida y guardada en: {out_path}")

    def show_final_images(self, original_path, segmented_path):
        # Leer imágenes
        original = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
        segmented = cv2.cvtColor(cv2.imread(segmented_path), cv2.COLOR_BGR2RGB)

        # Redimensionar si es muy grande
        max_size = 600  # Tamaño máximo permitido
        scale = min(max_size / original.shape[1], max_size / original.shape[0])
        if scale < 1:
            original = cv2.resize(original, None, fx=scale, fy=scale)
            segmented = cv2.resize(segmented, None, fx=scale, fy=scale)

        # Convertir a formato compatible con Tkinter
        img_original = ImageTk.PhotoImage(Image.fromarray(original))
        img_segmented = ImageTk.PhotoImage(Image.fromarray(segmented))

        # Actualizar paneles
        self.panel_original.configure(image=img_original)
        self.panel_original.image = img_original

        self.panel_segmented.configure(image=img_segmented)
        self.panel_segmented.image = img_segmented


# Iniciar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentAppGUI(root)
    root.mainloop()

