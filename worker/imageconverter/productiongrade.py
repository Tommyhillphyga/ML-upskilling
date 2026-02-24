import os
import threading
import multiprocessing as mp
from queue import Queue
from typing import List
from PIL import Image
import time
import timeit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_DIR = "Screenshots"
OUTPUT_DIR = "results"

READ_THREADS = 4
WRITE_THREADS = 4
PROCESS_WORKERS = os.cpu_count()

READ_QUEUE_SIZE = 20
WRITE_QUEUE_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

SENTINEL = None


# -------------------------------------------------
# Utility: list image paths
# -------------------------------------------------
def list_images(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


# -------------------------------------------------
# Stage 1: READ (I/O bound → threads)
# -------------------------------------------------
def reader_worker(paths, read_queue: Queue):
    """
    Reads file paths and pushes into queue.
    """
    for path in paths:
        read_queue.put(path)

    # Signal end to processing stage
    for _ in range(PROCESS_WORKERS):
        read_queue.put(SENTINEL)


# -------------------------------------------------
# Stage 2: PROCESS (CPU bound → multiprocessing)
# -------------------------------------------------
def process_worker(read_queue: Queue, write_queue: Queue):
    """
    Converts image to black & white.
    Runs in separate processes.
    """
    while True:
        path = read_queue.get()

        if path is SENTINEL:
            write_queue.put(SENTINEL)
            break

        try:
            img = Image.open(path)
            bw = img.convert("1")

            write_queue.put((path, bw))

        except Exception as e:
            logger.error(f"Processing failed: {path} {e}")


# -------------------------------------------------
# Stage 3: WRITE (I/O bound → threads)
# -------------------------------------------------
def writer_worker(write_queue: Queue):
    """
    Writes images to disk.
    """
    while True:
        item = write_queue.get()

        if item is SENTINEL:
            break

        path, img = item

        try:
            filename = os.path.basename(path)
            out_path = os.path.join(OUTPUT_DIR, filename)

            img.save(out_path)

        except Exception as e:
            logger.error(f"Write failed: {path} {e}")


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_pipeline():
    start = time.time()

    paths = list_images(INPUT_DIR)
    logger.info(f"Total images: {len(paths)}")

    # Thread queues (bounded)
    read_queue = mp.Queue(maxsize=READ_QUEUE_SIZE)
    write_queue = mp.Queue(maxsize=WRITE_QUEUE_SIZE)

    # -------------------------
    # Start reader thread
    # -------------------------
    reader = threading.Thread(
        target=reader_worker,
        args=(paths, read_queue),
        daemon=True
    )
    reader.start()

    # -------------------------
    # Start process workers
    # -------------------------
    processes = []
    for _ in range(PROCESS_WORKERS):
        p = mp.Process(
            target=process_worker,
            args=(read_queue, write_queue),
            daemon=True
        )
        p.start()
        processes.append(p)

    # -------------------------
    # Start writer threads
    # -------------------------
    writers = []
    for _ in range(WRITE_THREADS):
        t = threading.Thread(
            target=writer_worker,
            args=(write_queue,),
            daemon=True
        )
        t.start()
        writers.append(t)

    # -------------------------
    # Join processes
    # -------------------------
    for p in processes:
        p.join()

    # Writers need sentinel signals
    for _ in range(WRITE_THREADS):
        write_queue.put(SENTINEL)

    for t in writers:
        t.join()

    logger.info(f"Done in {time.time() - start:.2f}s")


# -------------------------------------------------
# Entry
# -------------------------------------------------
if __name__ == "__main__":
    t1 = timeit.default_timer()
    run_pipeline()
    print(f"total time taken to complete is {timeit.default_timer()-t1}")


