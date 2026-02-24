import os
import time
import timeit
import threading
import queue
from typing import Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
INPUT_DIR = "Screenshots"
OUTPUT_DIR = "results"

READ_THREADS = 4
WRITE_THREADS = 4
PROCESS_WORKERS = os.cpu_count()

READ_QUEUE_SIZE = 20
WRITE_QUEUE_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

SENTINEL = None


# ---------------------------------------------------
# Generator: stream file paths from disk
# ---------------------------------------------------
def image_paths(folder: str) -> Iterator[str]:
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            yield os.path.join(folder, f)


# ---------------------------------------------------
# Stage 1 — READ (I/O bound)
# ---------------------------------------------------
def read_worker(path_iter, read_queue: queue.Queue):
    """
    Reads image paths and pushes them to read_queue.
    """
    for path in path_iter:
        read_queue.put(path)

    # Signal processing stage to stop
    for _ in range(PROCESS_WORKERS):
        read_queue.put(SENTINEL)


# ---------------------------------------------------
# Stage 2 — PROCESS (CPU bound)
# IMPORTANT:
# Load image inside process to avoid serialization cost.
# ---------------------------------------------------
def process_image(path: str) -> Tuple[str, Image.Image]:
    img = Image.open(path)
    bw = img.convert("1")
    return path, bw


def process_queue_worker(read_queue: queue.Queue,
                         write_queue: queue.Queue,
                         process_pool: ProcessPoolExecutor):
    """
    Pulls file paths from read_queue,
    submits CPU work to process pool,
    pushes results to write_queue.
    """
    futures = []

    while True:
        path = read_queue.get()

        if path is SENTINEL:
            break

        future = process_pool.submit(process_image, path)
        futures.append(future)

    # Collect results
    for f in futures:
        write_queue.put(f.result())

    # Signal writers
    for _ in range(WRITE_THREADS):
        write_queue.put(SENTINEL)


# ---------------------------------------------------
# Stage 3 — WRITE (I/O bound)
# ---------------------------------------------------
def write_worker(write_queue: queue.Queue):
    """
    Writes processed images to disk.
    """
    counter = 0

    while True:
        item = write_queue.get()

        if item is SENTINEL:
            break

        path, img = item

        filename = f"{counter}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)

        img.save(out_path)
        counter += 1


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def run_pipeline():
    start = time.time()

    read_queue = queue.Queue(maxsize=READ_QUEUE_SIZE)
    write_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)

    paths_iter = image_paths(INPUT_DIR)

    with ThreadPoolExecutor(max_workers=READ_THREADS) as read_pool, \
         ThreadPoolExecutor(max_workers=WRITE_THREADS) as write_pool, \
         ProcessPoolExecutor(max_workers=PROCESS_WORKERS) as process_pool:

        # ------------------------------------------
        # Stage 1: Reader thread
        # ------------------------------------------
        read_pool.submit(read_worker, paths_iter, read_queue)

        # ------------------------------------------
        # Stage 2: Processing dispatcher
        # (single thread controlling process pool)
        # ------------------------------------------
        dispatcher = threading.Thread(
            target=process_queue_worker,
            args=(read_queue, write_queue, process_pool),
            daemon=True
        )
        dispatcher.start()

        # ------------------------------------------
        # Stage 3: Writers
        # ------------------------------------------
        writers = [
            write_pool.submit(write_worker, write_queue)
            for _ in range(WRITE_THREADS)
        ]

        dispatcher.join()

        for w in writers:
            w.result()

    print(f"Pipeline finished in {time.time() - start:.2f}s")


# ---------------------------------------------------
# ENTRY
# ---------------------------------------------------
if __name__ == "__main__":
    t1 = timeit.default_timer()
    run_pipeline()
    print(f"total time taken to complete is {timeit.default_timer()-t1}")
