import streamlit as st
import numpy as np
import json
import random
import cv2
from io import BytesIO
from PIL import Image
import zipfile

def generate_canvas_from_json(json_path, canvas_size, n_objects):
    with open(json_path, 'r') as f:
        contour_data = json.load(f)

    canvas = np.ones(canvas_size, dtype=np.uint8)  # start with all pixels = 1

    all_keys = list(contour_data.keys())
    random.shuffle(all_keys)
    selected_keys = all_keys[:min(n_objects, len(all_keys))]

    contours_info = []
    for key in selected_keys:
        x_coords = contour_data[key]["x"]
        y_coords = contour_data[key]["y"]
        if len(x_coords) != len(y_coords):
            continue
        contour = np.array([[x, y] for x, y in zip(x_coords, y_coords)], dtype=np.int32).reshape(-1, 1, 2)
        contours_info.append((key, contour))

    for key, contour in contours_info:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = canvas.shape[1] - w
        max_y = canvas.shape[0] - h
        if max_x <= 0 or max_y <= 0:
            continue
        rand_x = random.randint(0, max_x)
        rand_y = random.randint(0, max_y)
        translated_contour = contour + np.array([[rand_x - x, rand_y - y]])
        cv2.drawContours(canvas, [translated_contour], -1, 0, -1)  # fill with 0

    return canvas

st.title("Microstructure Generator from JSON Contours")

width = st.sidebar.number_input("Width (px)", min_value=100, max_value=5000, value=512, step=10)
height = st.sidebar.number_input("Height (px)", min_value=100, max_value=5000, value=512, step=10)
n_objects = st.sidebar.number_input("Number of contours", min_value=1, max_value=1000, value=10, step=10)
json_path = "new_dataset_with_qd.json"

if st.sidebar.button("Generate Single Image"):
    with st.spinner("Generating single image..."):
        canvas = generate_canvas_from_json(json_path, (height, width), n_objects)
        img = np.uint8(canvas * 255)
        img = np.stack([img]*3, axis=-1)
        pil_img = Image.fromarray(img)
        st.image(pil_img, caption="Generated Microstructure", use_container_width=True)

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download PNG", data=byte_im, file_name="microstructure.png", mime="image/png")

# Novo botão para gerar múltiplas imagens
num_sections = st.sidebar.number_input("Number of sections to generate", min_value=1, max_value=10000, value=10, step=10)

if st.sidebar.button("Generate Multiple Sections"):
    with st.spinner(f"Generating {num_sections} images... This may take a while."):
        # Criar buffer ZIP em memória
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for i in range(num_sections):
                canvas = generate_canvas_from_json(json_path, (height, width), n_objects)
                img = np.uint8(canvas * 255)
                img = np.stack([img]*3, axis=-1)
                pil_img = Image.fromarray(img)

                img_bytes = BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                # Adiciona imagem ao ZIP com nome indexado
                zip_file.writestr(f"microstructure_{i+1:04d}.png", img_bytes.read())

        zip_buffer.seek(0)

        st.download_button(
            label=f"Download {num_sections} images ZIP",
            data=zip_buffer,
            file_name=f"microstructures_{num_sections}.zip",
            mime="application/zip"
        )
