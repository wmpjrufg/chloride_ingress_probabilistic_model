import json
import os
import random
import cv2
import shutil

import shapely as sh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def centroid_polygon(x: list, y: list) -> tuple[float, float, list]:
    """
    Calculate the centroid of a polygon defined by its vertices.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices

    :return: output[0] = x centroid, output[1] = y centroid and output[2] = coordinates of the polygon in tuple format
    """

    coords = []
    for i, j in zip(x, y):
        coords.append((i, j))
    polyg = sh.geometry.Polygon(coords)
    centre = polyg.centroid

    return centre.x, centre.y, coords


def transport_polygon(x: list, y: list, x_new: float, y_new: float) -> tuple[float, float]:
    """
    Translate a polygon defined by its vertices to a new position.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices
    :param x_new: new x-coordinate for the centroid of the polygon
    :param y_new: new y-coordinate for the centroid of the polygon

    :return: output[0] = new x-coordinates of the polygon vertices and output[1] = new y-coordinates of the polygon vertices
    """

    x_g, y_g, coords = centroid_polygon(x, y)
    dx = -x_g
    dy = -y_g

    # Translate the polygon to the origin
    transl_polygon_00 = sh.affinity.translate(sh.geometry.Polygon(coords), xoff=dx, yoff=dy)

    # Translate the polygon to the new position
    transl_polygon_xy = sh.affinity.translate(transl_polygon_00, xoff=x_new, yoff=y_new)

    # Extract the new x and y coordinates
    x_trans = [po[0] for po in transl_polygon_xy.exterior.coords]
    y_trans = [po[1] for po in transl_polygon_xy.exterior.coords]

    return x_trans, y_trans


def size_polygon(x: list, y: list) -> tuple[float, float]:
    """
    Calculate the size of a polygon defined by its vertices.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices

    :return: output[0] = length in x direction and output[1] = length in y direction
    """

    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    l_x = x_max - x_min
    l_y = y_max - y_min

    return l_x, l_y


def exclude_cont_bound(myconts: dict, n_x: float, n_y: float) -> dict:
    """
    Exclude contours that touch the boundary of the image.
    
    :param myconts: Dictionary of contours
    :param n_x: Width of the image or criterion for exclusion
    :param n_y: Height of the image or criterion for exclusion

    :return: Filtered dictionary of contours that do not touch the boundary
    """
    
    myconts_copy = myconts.copy()
    for i in myconts:
        i_contour = myconts[i]
        for j in range(len(i_contour)):
            x, y = i_contour[j][0]
            if x == 0 or y == 0 or x >= n_x or y >= n_y:
                save_or_not_save = False
                break
        else:
            save_or_not_save = True
        if save_or_not_save == False:
          del myconts_copy[i]
          
    return myconts_copy


def extract_contours(contours: dict) -> dict: 
    """
    Extract contours from a list of contours and return them as a dictionary.
    
    :param contours: Contours to be processed
    
    :return: Contour points. Each index of the dictionary corresponds to a contour.
    """

    myconts = {}
    for i, contour in enumerate(contours):
        myconts[str(i)] = np.array(contour)
        
    return myconts


def process_images_to_json(filepath: str, output_json: str, width: int = 2500, height: int = 2500) -> None:
    """
    Process images to extract contours and save them in a JSON file.

    :param filepath: Path to the image or directory containing images
    :param output_json: Path to the output JSON file
    :param width: Width of the images (default is 2500)
    :param height: Height of the images (default is 2500)
    """

    contours_json = {}
    image_area = width * height 

    if os.path.isdir(filepath):
        files = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif isinstance(filepath, (list, tuple)):
        files = filepath 
    else:
        files = [filepath] 

    # Read each image and process contours
    for filename in files:
        img = cv2.imread(filename)
        if img is None:
            print(f"Error to read image: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get contour
        myconts = extract_contours(contours)

        # Exclude contours that touch the boundary
        myc = exclude_cont_bound(myconts, width-1, height-1)

        # Create a dictionary to save contours in JSON format
        image_key = os.path.basename(filename)
        contours_json[image_key] = {}

        # Image background size to aggregate-mortar
        l_xmax = []
        l_ymax = []
        for idx, contour in myc.items():
            coords = contour[:, 0, :]
            x_coords = coords[:, 0].tolist()
            y_coords = coords[:, 1].tolist()
            res = size_polygon(x_coords, y_coords)
            l_xmax.append(res[0])
            l_ymax.append(res[1])
        l_xback = max(l_xmax)
        l_yback = max(l_ymax)
        l_max = max([l_xback, l_yback])
        # Colocar o tamanho do background como o tamanho do maior contorno e img na base 2
        
        # Save contours and their areas in the JSON format
        for idx, contour in myc.items():
            coords = contour[:, 0, :]
            x_coords = coords[:, 0].tolist()
            y_coords = coords[:, 1].tolist()
            xg, yg, _ = centroid_polygon(x_coords, y_coords)
            # x_coords_trans, y_coords_trans = transport_polygon(x_coords, y_coords, l_max/2, l_max/2)
            area = cv2.contourArea(contour)
            q = float(area / image_area)
            contours_json[image_key][f"{idx}"] = {
                'x': x_coords,
                'y': y_coords,
                'q': q,
                'lmax': l_max,
                'xg': xg,
                'yg': yg
            }
    with open(output_json, 'w') as f:
        json.dump(contours_json, f, indent=4)
    print(f"Contours extracted and saved to {output_json}")


def plot_contours_from_json(json_file: str, keys_to_plot: list = None, width: int = 2500, height: int = 2500) -> None:
    """
    Plot contours from a JSON file in a figure with 3 subplots:
    - Left: all contours filled (random one in yellow)
    - Center: single random contour filled (in yellow)
    - Right: random contour centered in square patch

    :param json_file: Path to the JSON file containing contours
    :param keys_to_plot: List of keys (image names) to plot. If None, plot all keys
    :param width: Width of the images (default is 2500)
    :param height: Height of the images (default is 2500)
    """

    with open(json_file, 'r') as f:
        contours_data = json.load(f)

    keys = keys_to_plot if keys_to_plot else contours_data.keys()

    for image_name in keys:
        if image_name not in contours_data:
            print(f"Warning: {image_name} not found in contours data!")
            continue

        contours = contours_data[image_name]
        img_all = np.zeros((height, width, 3), dtype=np.uint8)  # RGB
        img_random = np.zeros((height, width, 3), dtype=np.uint8)

        contour_list = []
        q_values = []
        x_list = []
        y_list = []

        for contour in contours.values():
            x = np.array(contour["x"])
            y = np.array(contour["y"])
            pts = np.vstack((x, y)).T.astype(np.int32).reshape((-1, 1, 2))
            contour_list.append(pts)
            q_values.append(contour.get("q", 0))
            x_list.append(x.tolist())
            y_list.append(y.tolist())

        # Escolher contorno aleatório
        if not contour_list:
            print(f"Warning: No contours to display in {image_name}")
            continue

        random_idx = random.randint(0, len(contour_list) - 1)
        random_contour = contour_list[random_idx]
        q_random = q_values[random_idx]
        x_rand = x_list[random_idx]
        y_rand = y_list[random_idx]

        # Plot todos os contornos em cinza, o aleatório em amarelo
        for i, c in enumerate(contour_list):
            color = (255, 255, 0) if i == random_idx else (255, 255, 255)  # amarelo ou branco
            cv2.drawContours(img_all, [c], -1, color, thickness=cv2.FILLED)

        # Plot contorno aleatório em amarelo na imagem vazia
        cv2.drawContours(img_random, [random_contour], -1, (255, 255, 0), thickness=cv2.FILLED)

        # Criar patch centralizado
        canvas_size = 512
        patch_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        x_trans, y_trans = transport_polygon(x_rand, y_rand, canvas_size // 2, canvas_size // 2)
        pts_centered = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(patch_img, [pts_centered], -1, (255, 255, 0), thickness=cv2.FILLED)

        # Plot com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(img_all)
        axes[0].set_title(f"All Contours - {image_name}")
        axes[0].axis('off')

        axes[1].imshow(img_random)
        axes[1].set_title(f"Random Contour - q = {q_random:.6f}")
        axes[1].axis('off')

        axes[2].imshow(patch_img)
        axes[2].set_title("Centralized Patch")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def crop_contours(json_file: str, output_dir: str, canvas_size: int = 512) -> None:
    """
    Cria imagens com contornos centrados (de fato), transladando o contorno
    para o centro da imagem antes de desenhar.
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, 'r') as f:
        contours_data = json.load(f)

    for image_name, contours in contours_data.items():
        for idx, contour in contours.items():
            x_coords = contour["x"]
            y_coords = contour["y"]
            l_max = contour["lmax"]

            # Transladar para centro da nova imagem
            x_new, y_new = canvas_size // 2, canvas_size // 2
            x_trans, y_trans = transport_polygon(x_coords, y_coords, x_new, y_new)

            # Criar imagem vazia
            blank = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

            # Criar e desenhar contorno centralizado
            pts = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(blank, [pts], -1, color=255, thickness=cv2.FILLED)

            # Crop final (opcional, mas aqui a imagem já está centrada)
            cropped = blank

            output_name = f"{os.path.splitext(image_name)[0]}_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, output_name), cropped)

    print(f"Contours cropped and saved to {output_dir}.")


def flatten_contours_json(original_json_path: str, output_json_path: str) -> None:
    """
    Converte um JSON com estrutura por imagem em um dicionário plano com chave única para cada contorno.
    """

    with open(original_json_path, 'r') as f:
        data = json.load(f)

    flat_data = {}

    for image_name, contours in data.items():
        base_name = os.path.splitext(image_name)[0]  # remove .png
        for idx, contour in contours.items():
            key = f"{base_name}_{idx}.png"
            # Apenas x, y e q
            flat_data[key] = {
                "x": contour.get("x", []),
                "y": contour.get("y", []),
                "q": contour.get("q", None)
            }

    with open(output_json_path, 'w') as f:
        json.dump(flat_data, f, indent=4)

    print(f"New flattened JSON saved to {output_json_path} with {len(flat_data)} contours.")


def generate_dataset_csv_from_real_mask(
    flat_json_path: str,
    image_dir: str,
    output_csv_path: str,
    px_to_mm: float = 3.0 / 100.0  # 1 px = 0.03 mm
) -> None:
    """
    Gera um CSV com colunas: image_name, qd (área relativa), d (diâmetro real em mm)
    usando a imagem binária real para calcular o diâmetro via círculo mínimo.

    :param flat_json_path: Caminho para o JSON achatado (com q)
    :param image_dir: Diretório com as imagens centralizadas (ex: image_patchs)
    :param output_csv_path: Caminho para salvar o CSV de saída
    :param px_to_mm: Fator de conversão de pixel para milímetro (default: 0.03)
    """

    with open(flat_json_path, 'r') as f:
        data = json.load(f)

    records = []

    for name, values in data.items():
        image_name = name if name.endswith(".png") else f"{name}.png"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Read binary image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contour found in {image_name}")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(largest_contour)
        diam_px = 2 * radius
        diam_mm = diam_px * px_to_mm

        q = values.get("q", None)
        if q is None:
            continue

        records.append({
            "image_name": image_name,
            "qp": q,
            "qd": diam_mm
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved to: {output_csv_path} with {len(df)} samples.")


def filter_images_by_diameter(
    csv_path: str,
    image_dir: str,
    output_dir: str = "patchs_filtered",
    threshold_diam_mm: float = 15.0
) -> None:
    """
    Lê um CSV contendo os nomes das imagens e seus diâmetros (em mm),
    e copia para uma nova pasta apenas as imagens cujo diâmetro é menor que o limiar especificado.

    :param csv_path: Caminho para o CSV com colunas ['image', 'd'] (diâmetro em mm)
    :param image_dir: Diretório onde estão localizadas as imagens originais
    :param output_dir: Diretório de destino para imagens filtradas (default: 'patchs_filtered')
    :param threshold_diam_mm: Limiar de diâmetro para filtragem (default: 15.0 mm)
    """
    df = pd.read_csv(csv_path)

    if 'image_name' not in df.columns or 'qd' not in df.columns:
        raise ValueError("The CSV must contain 'image_name' and 'qd' columns.")

    os.makedirs(output_dir, exist_ok=True)

    count_copied = 0
    count_removed = 0

    for _, row in df.iterrows():
        image_name = row['image_name']
        d_mm = row['qd']

        if pd.isna(d_mm):
            continue

        if d_mm < threshold_diam_mm:
            src_path = os.path.join(image_dir, image_name)
            dst_path = os.path.join(output_dir, image_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                count_copied += 1
            else:
                print(f"Image not found: {src_path}")
        else:
            count_removed += 1

    print(f"\nImages with diameter < {threshold_diam_mm} mm copied to: {output_dir}")
    print(f"Total images copied: {count_copied}")
    print(f"Total images removed (>= {threshold_diam_mm} mm): {count_removed}")
