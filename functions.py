"""Functions to resolve paper: ..."""
import json
import os
import random
import shutil
from typing import Sequence
import zipfile
import time as ti

import cv2
import shapely as sh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import gdown



def transform_contours_dict(contours: Sequence) -> dict: 
    """
    Extract contours from a list of contours and return them as a dictionary.

    :param contours: Opencv contours to be processed

    :return: Contour points. Each index of the dictionary corresponds to a contour
    """

    myconts = {}
    for i, contour in enumerate(contours):
        myconts[str(i)] = np.array(contour)

    return myconts


def exclude_contours_in_boundary(myconts: dict, n_x: float, n_y: float) -> dict:
    """
    Exclude contours that touch the boundary of the image.
    
    :param myconts: Contour points. Each index of the dictionary corresponds to a contour
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


def size_polygon(x: list, y: list) -> tuple[float, float]:
    """
    Calculate the size of a polygon defined by its vertices.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices

    :return: [0] = length in x direction, [1] = length in y direction
    """

    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    l_x = x_max - x_min
    l_y = y_max - y_min

    return l_x, l_y


def centroid_polygon(x: list, y: list) -> tuple[float, float, list]:
    """
    Calculate the centroid of a polygon defined by its vertices.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices

    :return: [0] = x centroid, [1] = y centroid, [2] = coordinates of the polygon in tuple format
    """

    coords = []
    for i, j in zip(x, y):
        coords.append((i, j))
    polyg = sh.geometry.Polygon(coords)
    centre = polyg.centroid

    return centre.x, centre.y, coords


def trans_polygon_to_x_y(x: list, y: list, d_x: float, d_y: float) -> tuple[list, list]:
    """
    Translate a polygon defined by its vertices to a new position.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices
    :param x_new: new x-coordinate for the 0,0 of the polygon
    :param y_new: new y-coordinate for the 0,0 of the polygon

    :return: [0] = new x-coordinates of the polygon vertices, [1] = new y-coordinates of the polygon vertices
    """

    coords = list(zip(x, y))
    polygon = sh.geometry.Polygon(coords)
    translated = sh.affinity.translate(polygon, xoff=d_x, yoff=d_y)
    x_trans = [p[0] for p in translated.exterior.coords]
    y_trans = [p[1] for p in translated.exterior.coords]

    return x_trans, y_trans


def trans_rota_polygon(x: list, y: list, x_new: float, y_new: float, angle: float = 0, originn: str = 'centroid') -> tuple[list, list]:
    """
    Translate a polygon defined by its vertices to a new position.

    :param x: x-coordinates of the polygon vertices
    :param y: y-coordinates of the polygon vertices
    :param x_new: new x-coordinate for the centroid of the polygon
    :param y_new: new y-coordinate for the centroid of the polygon
    :param angle: rotation angle in degrees (counter-clockwise)
    :param origin: point to rotate around ('centroid' or 'center')

    :return: [0] = new x-coordinates of the polygon vertices, [1] = new y-coordinates of the polygon vertices
    """

    coords = list(zip(x, y))
    polygon = sh.geometry.Polygon(coords)
    x_g, y_g = polygon.centroid.coords[0]
    translated = sh.affinity.translate(polygon, xoff=-x_g, yoff=-y_g)
    rotated = sh.affinity.rotate(translated, angle, origin=originn)
    transformed = sh.affinity.translate(rotated, xoff=x_new, yoff=y_new)
    x_trans = [p[0] for p in transformed.exterior.coords]
    y_trans = [p[1] for p in transformed.exterior.coords]
    
    return x_trans, y_trans


def crop_contours(json_file: str, output_dir: str, canvas_size: int = 512) -> None:
    """
    Create cropped images from contours defined in a JSON file.

    :param json_file: Path to the JSON file containing contours by image
    :param output_dir: Directory to save the cropped images
    :param canvas_size: Size of the canvas for cropping (default is 512 px)
    """

    os.makedirs(output_dir, exist_ok=True)
    with open(json_file, 'r') as f:
        contours_data = json.load(f)

    for image_name, contours in contours_data.items():
        for idx, contour in contours.items():
            x_coords = contour["x coordinate in 0,0"]
            y_coords = contour["y coordinate in 0,0"]
            x_new, y_new = canvas_size // 2, canvas_size // 2
            x_trans, y_trans = trans_rota_polygon(x_coords, y_coords, x_new, y_new)
            blank = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            pts = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(blank, [pts], -1, color=255, thickness=cv2.FILLED)
            cropped = blank
            output_name = f"{os.path.splitext(image_name)[0]}_{idx}.png"
            cv2.imwrite(os.path.join(output_dir, output_name), cropped)

    print(f"Contours cropped and saved to {output_dir}.")


def generate_dataset_csv_from_real_mask(flat_json_path: str, image_dir: str = 'dataset/binary_patchs', output_csv_path: str = 'dataset_contours_aggregate_by_patch.csv', px_to_mm: float = 3.0 / 100.0) -> None:
    """
    Generate a CSV file with columns: image_name, qd (relative area), d (real diameter in mm)

    :param flat_json_path: Path to the flattened JSON by patch (with q)
    :param image_dir: Directory with the centered images
    :param output_csv_path: Path to save the output CSV
    :param px_to_mm: Pixel to millimeter conversion factor (default: 0.03)
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

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Diameter
        largest_contour = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(largest_contour)
        diam_px = 2 * radius
        diam_mm = diam_px * px_to_mm

        records.append({
            'image_name': image_name,
            'area (px)': values['contour area (px)'],
            'area (mm2)': (75*75) * values['contour area (px)'] / (2500*2500),
            'diameter (px)': diam_px,
            'diameter (mm)': diam_mm
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Contours by file extracted and saved to {output_csv_path} with {len(df)} samples.")


def process_images_to_json(filepath: str, name_output_json: str, output_path_to_patchs: str, width: int=2500, height: int=2500) -> tuple[int, int]:
    """
    Process images to extract contours and save them in a JSON file. This function save one json file per image, one json file per contour.

    :param filepath: Path to the images or path image
    :param name_output_json: Name to the output JSON file without extension
    :param output_path_to_patchs: Directory to save the cropped images
    :param width: Width of the images (default is 2500 px)
    :param height: Height of the images (default is 2500 px)

    :return: [0] = Number of images processed, output[1] = Number of contours extracted and saved in the JSON file
    """

    # Variables
    image_area = width * height 
    image_patch = 512 * 512
    boundary_condition_w = width - 2
    boundary_condition_h = height - 2

    # Get images
    contours_json = {}
    if os.path.isdir(filepath):
        files = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files_path = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

        # Get contour
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        myconts = transform_contours_dict(contours)

        # Exclude contours that touch the boundary
        myc_filtered = exclude_contours_in_boundary(myconts, boundary_condition_w, boundary_condition_h)

        # Create a dictionary to save contours in JSON format
        image_key = os.path.basename(filename)
        contours_json[image_key] = {}

        # Calculate the maximum size of the contours
        l_xmax = []
        l_ymax = []
        for idx, contour in myc_filtered.items():
            coords = contour[:, 0, :]
            x_coords = coords[:, 0].tolist()
            y_coords = coords[:, 1].tolist()
            res = size_polygon(x_coords, y_coords)
            l_xmax.append(res[0])
            l_ymax.append(res[1])
        l_xback = max(l_xmax)
        l_yback = max(l_ymax)
        l_max = max([l_xback, l_yback])
        
        # Save contours and their areas in the JSON format
        for idx, contour in myc_filtered.items():
            coords = contour[:, 0, :]
            x_coords = coords[:, 0].tolist()
            y_coords = coords[:, 1].tolist()
            x_00, y_00 = trans_rota_polygon(x_coords, y_coords, 0 , 0)
            area = cv2.contourArea(contour)
            q_ga = area
            q_pic = float(area / image_area)
            q_pat = float(area / image_patch)
            contours_json[image_key][f"{idx}"] = {
                'x coordinate in original image': x_coords,
                'y coordinate in original image': y_coords,
                'contour area (px)': q_ga,
                'rate of contour area to image area': q_pic,
                'rate of contour area to patch area': q_pat,
                'major width extract to contours': l_max,
                'x coordinate in 0,0': x_00,
                'y coordinate in 0,0': y_00
            }
        
    # Write the contours to a JSON file
    new_output_json = name_output_json
    name_output_json += '_by_image.json'
    with open(name_output_json, 'w') as f:
        json.dump(contours_json, f, indent=4)
        print(f"Contours by file extracted and saved to {name_output_json}")
    crop_contours(name_output_json, output_path_to_patchs)
    new_output_json += '_by_patch.json'
    with open(name_output_json, 'r') as f:
        data = json.load(f)
    flat_data = {}
    for image_name, contours in data.items():
        base_name = os.path.splitext(image_name)[0]
        for idx, contour in contours.items():
            key = f"{base_name}_{idx}.png"
            flat_data[key] = {
                'x coordinate in 0,0': contour['x coordinate in 0,0'],
                'y coordinate in 0,0': contour['y coordinate in 0,0'],
                'rate of contour area to image area': contour['rate of contour area to image area'],
                'rate of contour area to patch area': contour['rate of contour area to patch area'],
                'contour area (px)': contour['contour area (px)']
            }
    with open(new_output_json, 'w') as f:
        json.dump(flat_data, f, indent=4)
        print(f"Contours by file extracted and saved to {new_output_json}")

    # Write in csv file using diameter an area information
    generate_dataset_csv_from_real_mask(new_output_json)

    return len(files_path), len(flat_data)


def plot_contours_from_json(json_file: str, keys_to_plot: list, width: int = 2500, height: int = 2500) -> None:
    """
    Plot contours from a JSON file in a figure with 3 subplots:
    - Left: all contours filled (random one in yellow)
    - Center: single random contour filled (in yellow)
    - Right: random contour centered in square patch

    :param json_file: Path to the JSON file containing contours
    :param keys_to_plot: Keys (image names) to plot.
    :param width: Width of the images (default is 2500)
    :param height: Height of the images (default is 2500)
    """

    with open(json_file, 'r') as f:
        contours_data = json.load(f)

    for image_name in keys_to_plot:
        if image_name not in contours_data:
            print(f"Warning: {image_name} not found in contours data!")
            continue

        contours = contours_data[image_name]
        img_all = np.zeros((height, width, 3), dtype=np.uint8)
        img_random = np.zeros((height, width, 3), dtype=np.uint8)

        contour_list = []
        q_values = []
        x_list = []
        y_list = []

        for contour in contours.values():
            x = np.array(contour['x coordinate in original image'])
            y = np.array(contour['y coordinate in original image'])
            pts = np.vstack((x, y)).T.astype(np.int32).reshape((-1, 1, 2))
            contour_list.append(pts)
            q_values.append(contour['contour area (px)'])
            x_list.append(x.tolist())
            y_list.append(y.tolist())

        # Random contour to plot
        random_idx = random.randint(0, len(contour_list) - 1)
        random_contour = contour_list[random_idx]
        q_random = q_values[random_idx]
        x_rand = x_list[random_idx]
        y_rand = y_list[random_idx]

        # Plot all contours, one contour is plot using yellow color
        for i, c in enumerate(contour_list):
            color = (255, 255, 0) if i == random_idx else (255, 255, 255)  # amarelo ou branco
            cv2.drawContours(img_all, [c], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(img_random, [random_contour], -1, (255, 255, 0), thickness=cv2.FILLED)

        # Patch
        canvas_size = 512
        patch_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        x_trans, y_trans = trans_rota_polygon(x_rand, y_rand, canvas_size // 2, canvas_size // 2)
        pts_centered = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(patch_img, [pts_centered], -1, (255, 255, 0), thickness=cv2.FILLED)

        # Plot com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(img_all)
        axes[0].set_title(f"Image number: {image_name}")
        axes[0].axis('off')

        axes[1].imshow(img_random)
        axes[1].set_title("Position in image")
        axes[1].axis('off')

        axes[2].imshow(patch_img)
        axes[2].set_title(f"Patch: {q_random:.2f} px²")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def filter_images_by_diameter(csv_path: str = "dataset_contours_aggregate_by_patch.csv", image_dir: str = "dataset/binary_patchs", output_dir: str = "dataset/binary_patchs_filtered", contours_json_path: str = "dataset_contours_aggregate_by_patch.json", threshold_diam_mm: float = 15.0) -> None:
    """
    Read a set of images and delete all the images whose diameter is greater than to the specified threshold.

    :param csv_path: Path to the CSV file with columns
    :param image_dir: Directory where the original images are located
    :param output_dir: Destination directory for filtered images
    :param threshold_diam_mm: Diameter threshold for filtering (default: 15.0 mm)
    """

    # Load CSV
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Filter by diameter
    df_filtered = df[df['diameter (mm)'] < threshold_diam_mm]
    filtered_image_names = set(df_filtered['image_name'].tolist())

    n_total_images = len(df)
    n_filtered_images = len(df_filtered)

    # Copy filtered images
    for image_name in filtered_image_names:
        src_path = os.path.join(image_dir, image_name)
        dst_path = os.path.join(output_dir, image_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    # Save filtered CSV
    csv_filtered_path = os.path.splitext(csv_path)[0] + "_filtered.csv"
    df_filtered.to_csv(csv_filtered_path, index=False)

    # Load full contours JSON
    with open(contours_json_path, "r") as f:
        contours_data = json.load(f)

    # Filter contours JSON to keep only keys in filtered_image_names
    filtered_contours = {k: v for k, v in contours_data.items() if k in filtered_image_names}

    # Save filtered contours JSON
    json_filtered_path = os.path.splitext(contours_json_path)[0] + "_filtered.json"
    with open(json_filtered_path, "w") as f:
        json.dump(filtered_contours, f, indent=4)

    print(f"\nImages with diameter < {threshold_diam_mm} mm copied to: {output_dir}")
    print(f"Total images copied: {n_filtered_images}")
    print(f"Total images removed (>= {threshold_diam_mm} mm): {n_total_images - n_filtered_images}")
    print(f"Filtered CSV saved to: {csv_filtered_path}")
    print(f"Filtered contours JSON saved to: {json_filtered_path}")


def sort_contours_using_uniform_pdf_and_group(csv_path: str, json_path: str, n_objects: int, n_groups: int = 20):
    """
    Sort contours using a uniform probability density function and group them. 

    :param csv_path: Path to the CSV file with contour data
    :param json_path: Path to the JSON file with contour data
    :param n_objects: Number of objects to sample
    :param n_groups: Number of groups to create

    :return: Sorted and grouped contours
    """

    # Carrega o JSON e transforma em DataFrame
    with open(json_path, 'r') as f:
        contour_data = json.load(f)
    df_json = pd.DataFrame.from_dict(contour_data, orient='index')
    df_json['image_name'] = df_json.index
    df_json = df_json.reset_index(drop=True)

    # Carrega o CSV
    df_csv = pd.read_csv(csv_path)

    # Faz o merge dos dois
    df_full = pd.merge(df_json, df_csv, on='image_name')
    df_full = df_full[['image_name', 'x coordinate in 0,0', 'y coordinate in 0,0',
                       'diameter (px)', 'diameter (mm)', 'area (px)', 'area (mm2)']]

    # Define se precisa repetir amostras
    replace_flag = n_objects > len(df_full)

    # Amostragem (com repetição se necessário)
    df_selected_contours = df_full.sample(n=n_objects, replace=replace_flag)

    # Ordena pelo diâmetro e cria grupos
    df_sorted = df_selected_contours.sort_values('diameter (px)', ascending=False).reset_index(drop=True)
    group_dim = np.array_split(df_sorted.index, n_groups)
    df_sorted['group by diameter (px)'] = -1
    for i, group in enumerate(group_dim):
        df_sorted.loc[group, 'group by diameter (px)'] = i + 1

    return df_sorted


def generate_canvas_from_json(json_path, canvas_size, n_objects):
    """
    Generate a canvas image from contour data in a JSON file.

    :param json_path: Path to the JSON file containing contour data
    :param canvas_size: Size of the canvas (height, width)
    :param n_objects: Number of objects to include in the canvas
    """
    
    # Open json file and created 1,1,1,1 matrix
    with open(json_path, 'r') as f:
        contour_data = json.load(f)

    # Sorted keys using diameter column: decrease way
    all_keys = list(contour_data.keys())
    sampled_elements = list(np.random.choice(all_keys, size=n_objects, replace=False))
    sampled_elements.sort(key=lambda k: contour_data[k]["contour area (px)"], reverse=True)

    # Latim Hypercube Centroids
    sampler = scipy.stats.qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n_objects)
    scaled_sample = scipy.stats.qmc.scale(sample, l_bounds=[0, 0], u_bounds=[canvas_size[0], canvas_size[1]])

    # Separando coordenadas x (largura) e y (altura)
    x_centroids = scaled_sample[:, 0]
    y_centroids = scaled_sample[:, 1]
   
    contours_info = []
    # matriz_binaria = np.zeros(canvas_size, dtype=np.uint8)
    # for id, value in enumerate(sampled_elements):
    #     x_coords = contour_data[value]["x coordinate in 0,0"]
    #     y_coords = contour_data[value]["y coordinate in 0,0"]
    #     x_new, y_new = x_centroids[id], y_centroids[id]
    #     x_trans, y_trans = trans_rota_polygon(x_coords, y_coords, x_new, y_new)
    #     blank = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint8)
    #     pts = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
    #     cv2.drawContours(blank, [pts], -1, color=255, thickness=cv2.FILLED)
    #     cropped = blank
    #     matriz_binaria += cropped
    #     # matriz_binaria = (cropped == 255).astype(int)
    # np.savetxt("matriz_retangulo.txt", matriz_binaria, fmt='%d')
    # plt.imshow(matriz_binaria)# , cmap="gray")  # 'gray' para manter tons de cinza
    # plt.axis("off")                   # remove eixos
    # plt.show()

    matriz_contagem = np.zeros(canvas_size, dtype=np.uint16)
    for id, value in enumerate(sampled_elements):
        x_coords = contour_data[value]["x coordinate in 0,0"]
        y_coords = contour_data[value]["y coordinate in 0,0"]
        x_new, y_new = x_centroids[id], y_centroids[id]
        x_trans, y_trans = trans_rota_polygon(x_coords, y_coords, x_new, y_new)

        blank = np.zeros(canvas_size, dtype=np.uint8)
        pts = np.array(list(zip(x_trans, y_trans)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(blank, [pts], -1, color=255, thickness=cv2.FILLED)

        # máscara 0/1 e acumula em uint16
        mask = (blank == 255).astype(np.uint16)
        matriz_contagem += mask

    # Pixels com sobreposição (≥2 contornos)
    overlap_mask = (matriz_contagem >= 2)
    print("estou aqui: ", np.sum(overlap_mask), overlap_mask.shape)

    # Se quiser visualizar a contagem (sem estourar):
    plt.imshow(matriz_contagem, cmap="gray")
    plt.axis("off")
    plt.show()

    # # Se quiser salvar como texto (contagens inteiras):
    # np.savetxt("matriz_contagem.txt", matriz_contagem, fmt='%d')

    # for key, contour in contours_info:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     max_x = canvas.shape[1] - w
    #     max_y = canvas.shape[0] - h
    #     if max_x <= 0 or max_y <= 0:
    #         continue
    #     rand_x = random.randint(0, max_x)
    #     rand_y = random.randint(0, max_y)
    #     translated_contour = contour + np.array([[rand_x - x, rand_y - y]])
    #     cv2.drawContours(canvas, [translated_contour], -1, 0, -1)  # fill with 0


def obtain_cdf(x: list) -> tuple[list, list]:
    """
    Obtain the cumulative distribution function (CDF) of a list of values.

    :param x: values

    :return: [0] = sorted values, [1] = CDF of the input values
    """

    x_sorted = np.sort(x)
    x_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    return list(x_sorted), list(x_cdf)


def groups_within_radius(trees, point, r = 50):
    hits = []
    for gi, t in enumerate(trees):
        idxs = t.query_ball_point(point, r)
        if idxs:               # se houver ao menos um vizinho naquele grupo
            hits.append(gi)    # gi = índice do grupo em data_points
    return hits


def noise_point(y: list, value_noise: float = 1):
    """
    Apply noise to a list of values.

    :param y: input values
    :param value_noise: noise percentage. 0 to 100

    :return: Values with noise applied
    """

    noise_perc = [float(i)/100 for i in list(np.random.uniform(-value_noise, value_noise, size=len(y)))]
    y_noise = [i + i*j for i, j in zip(y, noise_perc)]

    return y_noise


def download_and_extract_gdrive_zip(file_id: str, output_zip: str = "file_downloaded.zip"):
    """
    Download a .zip file from Google Drive and extract it in the current folder.

    :param file_id: Google Drive file ID of the .zip file
    :param output_zip: Name of the .zip file to be saved locally
    """

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {output_zip}...")
    gdown.download(url, output_zip, quiet=False)
    print(f"Extracting {output_zip}...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction concluded!")


def normalize_contours(contours, drop_last_if_closed=True):
    """contours: list[list[tuple(x,y)]]. Retorna list[np.ndarray(N,2)]."""
    arrays = []
    for pts in contours:
        arr = np.asarray(pts, dtype=float).reshape(-1, 2)
        # remove o último ponto se for igual ao primeiro (contorno fechado)
        if drop_last_if_closed and len(arr) > 1 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        arrays.append(arr)
    return arrays


def generate_cross_section_w(x_list: list, y_list: list, n_s: int = 300, dataset_csv: str = "dataset_contours_aggregate_by_patch_filtered.csv", dataset_json: str = "dataset_contours_aggregate_by_patch_filtered.json") -> dict:
    """
    Generate a cross-section contour from the given parameters and save image and contour dataset in the specified output paths.

    :param x_list: X coordinates of the boundary
    :param y_list: Y coordinates of the boundary
    :param n_s: Number of samples. Default is 300
    :param dataset_csv: Path to the dataset CSV file. Default is "dataset_contours_aggregate_by_patch.csv"
    :param dataset_json: Path to the dataset JSON file. Default is "dataset_contours_aggregate_by_patch.json"
    # :param output_json: Path to the output JSON file. Default is "output_contour.json"
    # :param output_img: Path to the output image file. Default is "output_contour.png"

    :return: dataset with the cropped contours
    """
    # Load dataset contours
    df = sort_contours_using_uniform_pdf_and_group(dataset_csv,dataset_json, n_s)

    # Generate non-colliding contours
    contours = []
    
    for m, row in df.iterrows():
        sampler = sc.stats.qmc.LatinHypercube(d=2)
        centroids = sc.stats.qmc.scale(sampler.random(n=1), [x_list[0], y_list[0]], [x_list[2], y_list[2]]).squeeze()

        cx = noise_point([centroids[0]], value_noise=float(np.random.uniform(1, 2)))[0]
        cy = noise_point([centroids[1]], value_noise=float(np.random.uniform(1, 2)))[0]

        # Contour candidate
        x_new, y_new = trans_rota_polygon(row['x coordinate in 0,0'], row['y coordinate in 0,0'], cx, cy, angle=float(np.random.uniform(0, 360)))
        candidate = list(zip(x_new, y_new))

        # t0 = ti.perf_counter()
        if m == 0:
            contours.append(candidate)
            print(f"First contour added: {candidate}")
            print(f"Current contours: {contours}")
        else:
            cand_poly = sh.geometry.Polygon(candidate)
            collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours)
            tries = 0
            while collide and tries < 50:
                centroids = sc.stats.qmc.scale(sampler.random(n=1), [x_list[0], y_list[0]], [x_list[2], y_list[2]]).squeeze()
                cx = noise_point([centroids[0]], value_noise=float(np.random.uniform(1, 2)))[0]
                cy = noise_point([centroids[1]], value_noise=float(np.random.uniform(1, 2)))[0]
                x_new, y_new = trans_rota_polygon(row['x coordinate in 0,0'], row['y coordinate in 0,0'], cx, cy, angle=float(np.random.uniform(0, 360)))
                candidate = list(zip(x_new, y_new))
                cand_poly = sh.geometry.Polygon(candidate)
                collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours)
                tries += 1

            if not collide:
                contours.append(candidate)
        # print(f"Plotting took {ti.perf_counter() - t0:.2f} seconds")

    # Crop contours
    cropped_contours = []
    for cont in contours:
        xs, ys = zip(*cont)
        xs = np.array(xs)
        ys = np.array(ys)

        all_out_x = np.all((xs < x_list[0]) | (xs > x_list[2]))
        all_out_y = np.all((ys < y_list[0]) | (ys > y_list[2]))
        if all_out_x or all_out_y:
            continue

        xs_clipped = np.clip(xs, x_list[0], x_list[2])
        ys_clipped = np.clip(ys, y_list[0], y_list[2])

        cropped_contours.append(list(zip(xs_clipped, ys_clipped)))

    # # Boundary
    # boundary = [[x_list[0], 0], [x_list[2], 0], [x_list[2], y_list[2]], [x_list[0], y_list[2]], [x_list[0], 0]]

    data = {}
    for i, cont in enumerate(cropped_contours, start=1):
        xs, ys = zip(*cont)
        data[f"{i:02}"] = {
            "x coordinate": [float(x) for x in xs],
            "y coordinate": [float(y) for y in ys]
        }

    # bx, by = zip(*boundary)
    # data["boundary"] = {
    #     "x coordinate": [float(x) for x in bx],
    #     "y coordinate": [float(y) for y in by]
    # }

    with open('output_json.json', "w") as f:
        json.dump(data, f, indent=2)

    # Plot
    
    fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(np.zeros((H, W)), cmap="gray")

    for cont in cropped_contours:
        xs, ys = zip(*cont)
        ax.plot(xs, ys, color='blue', linewidth=1)
        ax.fill(xs, ys, color='blue', alpha=1)

    # bx, by = zip(*boundary)
    # ax.plot(bx, by, color='black', linewidth=2)

    # ax.set_xlim(0, img_w_px)
    # ax.set_ylim(img_h_px, 0)
    # ax.axis('off')

    # plt.savefig(output_img, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()

    return data


def generate_cross_section(x_list: list, y_list: list, n_s: int = 300, dataset_csv: str = "dataset_contours_aggregate_by_patch_filtered.csv", dataset_json: str = "dataset_contours_aggregate_by_patch_filtered.json") -> dict:
    """
    Generate a cross-section contour from the given parameters and save image and contour dataset in the specified output paths.

    :param x_list: X coordinates of the boundary
    :param y_list: Y coordinates of the boundary
    :param n_s: Number of samples. Default is 300
    :param dataset_csv: Path to the dataset CSV file. Default is "dataset_contours_aggregate_by_patch.csv"
    :param dataset_json: Path to the dataset JSON file. Default is "dataset_contours_aggregate_by_patch.json"
    # :param output_json: Path to the output JSON file. Default is "output_contour.json"
    # :param output_img: Path to the output image file. Default is "output_contour.png"

    :return: dataset with the cropped contours
    """
    # Load dataset contours
    df = sort_contours_using_uniform_pdf_and_group(dataset_csv,dataset_json, n_s)

    # Generate non-colliding contours
    contours = []
    
    for m, row in df.iterrows():
        sampler = sc.stats.qmc.LatinHypercube(d=2)
        centroids = sc.stats.qmc.scale(sampler.random(n=1), [x_list[0], y_list[0]], [x_list[2], y_list[2]]).squeeze()

        cx = noise_point([centroids[0]], value_noise=float(np.random.uniform(1, 2)))[0]
        cy = noise_point([centroids[1]], value_noise=float(np.random.uniform(1, 2)))[0]

        # Contour candidate
        x_new, y_new = trans_rota_polygon(row['x coordinate in 0,0'], row['y coordinate in 0,0'], cx, cy, angle=float(np.random.uniform(0, 360)))
        candidate = list(zip(x_new, y_new))

        # t0 = ti.perf_counter()
        if m == 0:
            contours.append(candidate)
        elif m > 0 and m < 10:
            cand_poly = sh.geometry.Polygon(candidate)
            collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours)
            tries = 0
            while collide and tries < 50:
                centroids = sc.stats.qmc.scale(sampler.random(n=1), [x_list[0], y_list[0]], [x_list[2], y_list[2]]).squeeze()
                cx = noise_point([centroids[0]], value_noise=float(np.random.uniform(1, 2)))[0]
                cy = noise_point([centroids[1]], value_noise=float(np.random.uniform(1, 2)))[0]
                x_new, y_new = trans_rota_polygon(row['x coordinate in 0,0'], row['y coordinate in 0,0'], cx, cy, angle=float(np.random.uniform(0, 360)))
                candidate = list(zip(x_new, y_new))
                cand_poly = sh.geometry.Polygon(candidate)
                collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours)
                tries += 1
        else:
            cand_poly = sh.geometry.Polygon(candidate)
            data_contours = normalize_contours(contours, drop_last_if_closed=True)
            trees = [sc.spatial.cKDTree(arr) for arr in data_contours if len(arr) > 0]
            point = np.array([cx, cy])
            ids = groups_within_radius(trees, point)
            contours_filtered = [c for i, c in enumerate(contours) if i in ids]
            collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours_filtered)
            tries = 0
            while collide and tries < 50:
                centroids = sc.stats.qmc.scale(sampler.random(n=1), [x_list[0], y_list[0]], [x_list[2], y_list[2]]).squeeze()
                cx = noise_point([centroids[0]], value_noise=float(np.random.uniform(1, 2)))[0]
                cy = noise_point([centroids[1]], value_noise=float(np.random.uniform(1, 2)))[0]
                x_new, y_new = trans_rota_polygon(row['x coordinate in 0,0'], row['y coordinate in 0,0'], cx, cy, angle=float(np.random.uniform(0, 360)))
                candidate = list(zip(x_new, y_new))
                cand_poly = sh.geometry.Polygon(candidate)
                collide = any(cand_poly.intersects(sh.geometry.Polygon(c)) for c in contours_filtered)
                tries += 1            

            if not collide:
                contours.append(candidate)
        # print(f"Plotting took {ti.perf_counter() - t0:.2f} seconds")

    # Crop contours
    cropped_contours = []
    for cont in contours:
        xs, ys = zip(*cont)
        xs = np.array(xs)
        ys = np.array(ys)

        all_out_x = np.all((xs < x_list[0]) | (xs > x_list[2]))
        all_out_y = np.all((ys < y_list[0]) | (ys > y_list[2]))
        if all_out_x or all_out_y:
            continue

        xs_clipped = np.clip(xs, x_list[0], x_list[2])
        ys_clipped = np.clip(ys, y_list[0], y_list[2])

        cropped_contours.append(list(zip(xs_clipped, ys_clipped)))

    # # Boundary
    # boundary = [[x_list[0], 0], [x_list[2], 0], [x_list[2], y_list[2]], [x_list[0], y_list[2]], [x_list[0], 0]]

    data = {}
    for i, cont in enumerate(cropped_contours, start=1):
        xs, ys = zip(*cont)
        data[f"{i:02}"] = {
            "x coordinate": [float(x) for x in xs],
            "y coordinate": [float(y) for y in ys]
        }

    # bx, by = zip(*boundary)
    # data["boundary"] = {
    #     "x coordinate": [float(x) for x in bx],
    #     "y coordinate": [float(y) for y in by]
    # }

    # with open('output_json.json', "w") as f:
    #     json.dump(data, f, indent=2)

    # Plot
    
    # fig, ax = plt.subplots(figsize=(7, 7))
    # # ax.imshow(np.zeros((H, W)), cmap="gray")

    # for cont in cropped_contours:
    #     xs, ys = zip(*cont)
    #     ax.plot(xs, ys, color='blue', linewidth=1)
    #     # ax.fill(xs, ys, color='blue', alpha=1)

    # # bx, by = zip(*boundary)
    # # ax.plot(bx, by, color='black', linewidth=2)

    # # ax.set_xlim(0, img_w_px)
    # # ax.set_ylim(img_h_px, 0)
    # # ax.axis('off')

    # # plt.savefig(output_img, dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.show()

    return data
