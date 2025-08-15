import json
import multiprocessing as mp
import time
from functions import generate_cross_section, trans_polygon_to_x_y

def process_rectangle(i, divs_b, divs_h, x_ini, y_ini, img_w_px, img_h_px):
    """
    Função que processa um único retângulo e retorna um dicionário de contornos.
    """
    dx = img_w_px * (i % divs_b)
    dy = img_h_px * (i // divs_b)

    print(f"[PID {mp.current_process().pid}] Processando retângulo {i+1}/{divs_b * divs_h} na posição ({dx:.2f}, {dy:.2f})")

    # Coordenadas do retângulo transladadas
    x, y = trans_polygon_to_x_y(x_ini, y_ini, dx, dy)
    del x[-1]
    del y[-1]

    # Gera contornos para esse retângulo
    co = generate_cross_section(x_list=x, y_list=y, n_s=400)

    return co


if __name__ == "__main__":
    # --- Parâmetros ---
    b = 150
    h = 300
    divs_b = 2
    divs_h = 2

    img_w_mm = b / divs_b
    img_h_mm = h / divs_h
    img_w_px = img_w_mm * 2500 / 75
    img_h_px = img_h_mm * 2500 / 75

    x_ini = [0, img_w_px, img_w_px, 0]
    y_ini = [0, 0, img_h_px, img_h_px]

    # --- Início da contagem de tempo ---
    start_time = time.perf_counter()

    num_cores = mp.cpu_count()
    print(f"\nUsando {num_cores} núcleos para processamento paralelo.")
    print("Iniciando processamento paralelo de contornos...")

    with mp.Pool(processes=num_cores) as pool:
        results = pool.starmap(
            process_rectangle,
            [(i, divs_b, divs_h, x_ini, y_ini, img_w_px, img_h_px) for i in range(divs_b * divs_h)]
        )

    # Junta todos os resultados
    all_contours = {}
    contour_id = 1
    for co in results:
        for _, contour in co.items():
            xs = contour["x coordinate"]
            ys = contour["y coordinate"]
            all_contours[f"{contour_id:02}"] = {
                "x coordinate": xs,
                "y coordinate": ys
            }
            contour_id += 1

    # --- Fim da contagem de tempo ---
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("\nGeração de contornos concluída!")
    print(f"Total de contornos gerados: {contour_id - 1}")
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos")

    # Salva JSON final
    with open("all_contours.json", "w") as f:
        json.dump(all_contours, f, indent=2)

    print("Arquivo JSON salvo com sucesso!")
