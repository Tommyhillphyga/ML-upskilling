import os
import timeit
from typing import Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image

INPUT_DIR = "Screenshots"
OUTPUT_DIR = "results"

READ_THREADS = 8
WRITE_THREADS = 8
PROCESS_WORKERS = os.cpu_count()


os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Iterator for image paths
# -------------------------
def image_paths(folder: str) -> Iterator[str]:
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            yield os.path.join(folder, f)


# -------------------------
# Stage 1 (I/O bound)
# Read image
# -------------------------
def read_image(path: str) -> Tuple[str, Image.Image]:
    img = Image.open(path)
    return path, img


# -------------------------
# Stage 2 (CPU bound)
# Convert to black & white
# IMPORTANT:
# Load again inside process to avoid
# serializing PIL objects.
# -------------------------
def convert_bw_process(path: str) -> Tuple[str, Image.Image]:
    img = Image.open(path)
    bw = img.convert("1")
    return path, bw


# -------------------------
# Stage 3 (I/O bound)
# Write image
# -------------------------
def write_image(index_img):
    index, img = index_img
    out_path = os.path.join(OUTPUT_DIR, f"{index}.png")
    img.save(out_path)


# -------------------------
# Optimized pipeline
# -------------------------
def optimized_pipeline():
    paths = list(image_paths(INPUT_DIR))

    print(f"Total images: {len(paths)}")

    # CPU stage (multiprocessing)
    with ProcessPoolExecutor(max_workers=PROCESS_WORKERS) as proc_pool:

        futures = [
            proc_pool.submit(convert_bw_process, path)
            for path in paths
        ]

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # I/O stage (multithreading)
    with ThreadPoolExecutor(max_workers=WRITE_THREADS) as thread_pool:
        indexed_results = [
            (i, img) for i, (_, img) in enumerate(results)
        ]
        thread_pool.map(write_image, indexed_results)


# -------------------------
# Runner
# -------------------------
if __name__ == "__main__":
    t0 = timeit.default_timer()
    optimized_pipeline()
    print("Time:", timeit.default_timer() - t0)