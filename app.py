import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile
import tempfile
from functions import generate_cross_section  # importa sua função

st.title("Cross Section Generator from Contours Dataset")

# Entradas do usuário
width_mm = st.sidebar.number_input("Width (mm)", min_value=10, max_value=1000, value=100, step=10)
height_mm = st.sidebar.number_input("Height (mm)", min_value=10, max_value=1000, value=100, step=10)
dataset_csv = "dataset_contours_aggregate_by_patch.csv"
dataset_json = "dataset_contours_aggregate_by_patch.json"

# Inicializa sessão
if "last_zip_bytes" not in st.session_state:
    st.session_state.last_zip_bytes = None

# ------------------ Geração unitária ------------------
if st.sidebar.button("Generate Single Image"):
    with st.spinner("Generating single cross section..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = f"{tmpdir}/temp_contour.json"
            output_img = f"{tmpdir}/temp_contour.png"

            generate_cross_section(width_mm, height_mm, dataset_csv, dataset_json, output_json, output_img)

            # Cria ZIP em memória
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.write(output_json, arcname="cross_section.json")
                zip_file.write(output_img, arcname="cross_section.png")

            zip_buffer.seek(0)
            st.session_state.last_zip_bytes = zip_buffer.getvalue()

# Exibe e disponibiliza download se houver ZIP gerado
if st.session_state.last_zip_bytes:
    with st.spinner("Preparing ZIP..."):
        # Mostrar a imagem dentro do ZIP
        with zipfile.ZipFile(BytesIO(st.session_state.last_zip_bytes), "r") as zf:
            with zf.open("cross_section.png") as f:
                img = Image.open(f)
                st.image(img, caption="Generated Cross Section", use_container_width=True)

        st.download_button(
            "Download ZIP (Image + JSON)",
            data=st.session_state.last_zip_bytes,
            file_name="cross_section.zip",
            mime="application/zip"
        )

# # ------------------ Geração múltipla ------------------
# num_sections = st.sidebar.number_input("Number of sections to generate", min_value=1, max_value=1000, value=10, step=1)

# if st.sidebar.button("Generate Multiple Sections"):
#     with st.spinner(f"Generating {num_sections} cross sections... This may take a while."):
#         with tempfile.TemporaryDirectory() as tmpdir:
#             zip_buffer = BytesIO()
#             with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
#                 for i in range(num_sections):
#                     output_json = f"{tmpdir}/temp_contour_{i}.json"
#                     output_img = f"{tmpdir}/temp_contour_{i}.png"

#                     generate_cross_section(width_mm, height_mm, dataset_csv, dataset_json, output_json, output_img)

#                     zip_file.write(output_json, arcname=f"cross_section_{i+1:04d}.json")
#                     zip_file.write(output_img, arcname=f"cross_section_{i+1:04d}.png")

#             zip_buffer.seek(0)
#             st.download_button(
#                 label=f"Download {num_sections} cross sections ZIP",
#                 data=zip_buffer,
#                 file_name=f"cross_sections_{num_sections}.zip",
#                 mime="application/zip"
#             )
