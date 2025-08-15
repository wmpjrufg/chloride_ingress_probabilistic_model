import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import json
from PIL import Image
from functions import generate_cross_section

# ------------------------------
# Parâmetros
# ------------------------------
divs = 4
final_px = 2500
mm_full = 75
px_per_mm = final_px / mm_full
tile_px = final_px // divs          # 625 px
tile_mm = tile_px / px_per_mm       # 18.75 mm

# ------------------------------
# Função de processamento do tile
# ------------------------------
def process_tile(args):
    i, j = args
    img_name = f"tile_{i}_{j}.png"
    json_name = f"tile_{i}_{j}.json"

    print(f"[Tile {i},{j}] Generating cross-section...")
    # Gera a subimagem e o JSON temporário
    generate_cross_section(
        img_w_mm=tile_mm,
        img_h_mm=tile_mm,
        n_s=300,
        output_json=json_name,
        output_img=img_name
    )
    print(f"[Tile {i},{j}] Generated {img_name} and {json_name}")

    # Redimensiona imagem
    tile_img = Image.open(img_name).convert("RGB")
    tile_img = tile_img.resize((tile_px, tile_px), Image.Resampling.LANCZOS)

    # Lê o JSON gerado
    with open(json_name, "r", encoding="utf-8") as f:
        tile_data = json.load(f)

    # Calcula offset para coordenadas globais
    offset_x = j * tile_px
    offset_y = i * tile_px

    # Ajusta coordenadas (ignorando "boundary")
    adjusted_data = {}
    for key, cont in tile_data.items():
        if key == "boundary":
            continue
        new_x = [x + offset_x for x in cont["x coordinate"]]
        new_y = [y + offset_y for y in cont["y coordinate"]]
        adjusted_data[key] = {
            "x coordinate": new_x,
            "y coordinate": new_y
        }

    print(f"[Tile {i},{j}] Finished processing")
    return (i, j, tile_img, adjusted_data)

# ------------------------------
# Execução paralela
# ------------------------------
if __name__ == "__main__":
    print("Starting tile processing...")
    coords = [(i, j) for i in range(divs) for j in range(divs)]

    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_tile, coords)

    print("All tiles processed. Combining results...")

    # Monta imagem final e JSON final
    final_img = Image.new("RGB", (final_px, final_px))
    final_json = {}

    contour_index = 1
    total_tiles = len(results)
    for count, (i, j, tile_img, adjusted_data) in enumerate(results, start=1):
        final_img.paste(tile_img, (j * tile_px, i * tile_px))
        for _, cont in adjusted_data.items():
            final_json[f"{contour_index:02}"] = cont
            contour_index += 1
        print(f"[Combining] Tile {i},{j} ({count}/{total_tiles}) merged")

    # Adiciona boundary global
    boundary = [
        [0, 0],
        [final_px, 0],
        [final_px, final_px],
        [0, final_px],
        [0, 0]
    ]
    bx, by = zip(*boundary)
    final_json["boundary"] = {
        "x coordinate": list(bx),
        "y coordinate": list(by)
    }

    # Salva arquivos finais
    final_img.save("final_cross_section.png")
    print("Final image saved as 'final_cross_section.png'")

    with open("final_cross_section.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    print("Final JSON saved as 'final_cross_section.json'")

    # Mostra imagem final
    print("Displaying final image...")
    plt.imshow(final_img)
    plt.axis("off")
    plt.show()
    print("Done!")
