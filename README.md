# How to Run the Microstructure Generator Streamlit App

This tutorial will guide you through running a simple Streamlit app that generates microstructure images using contour data stored in a JSON file.

---

## Prerequisites

* Python 3.7 or higher installed on your system.
* Basic familiarity with running Python scripts from the command line.
* Your JSON file containing the contours (e.g., `new_dataset_contours_aggregate_qd.json`).

---

## Step 1: Create a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment to isolate dependencies:

```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

---

## Step 2: Install Required Packages

Install Streamlit and OpenCV (and other dependencies) in your virtual environment using a requirements file:

```bash
pip install -r .\requirements.txt
```

Or install the packages individually:

```bash
pip install streamlit opencv-python-headless matplotlib pillow numpy
```

---

## Step 4: Run the Streamlit App

In your terminal or command prompt, run:

```bash
streamlit run app.py
```

This will start the Streamlit server and automatically open a browser window/tab with the app.

---

## Step 5: Use the App

* Use the sidebar to set the **width**, **height**, and **number of contours** to generate.
* Click the **"Generate Image"** button.
* Wait a few seconds while the app generates the microstructure.
* The generated image will be displayed.
* You can download the PNG image by clicking the **"Download PNG"** button.



