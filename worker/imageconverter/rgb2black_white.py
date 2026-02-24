# write a program to read multiple images from disk and covert them to black and white and write them back to disk. 
# implement the normal process, them optimise using iterator, multiprocessing and multithreading. 

# read the images - (i/o bound - multithreading)
# convert to black and white - (compute bound - multiprocessing)
# write it back to a disk (i/0 bound - multithreading)

import timeit
import queue 
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import os
import threading
from typing import Iterator, Tuple
import logging


logger = logging.getLogger(__name__)

THREAD_COUNT = 20

class CheckableQueue(queue.Queue):
    def __contains__(self, item):
        with self.mutex:
            return item in self.queue
        
    def __len__ (self):
        return len(self.queue)

class FrameLoader:
    """
    Iterates over image files in a directory (jpg, png, jpeg), returning (path, frame).
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"FrameLoader: directory not found: {path}")
        self.files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __iter__(self) -> Iterator[Tuple[str, any]]:
        for f in self.files:
            full = os.path.join(self.path, f)
            frame = Image.open(full)
            if frame is None:
                logger.warning("Failed to read frame: %s", full)
                continue
            yield frame

counter = 0
icp = CheckableQueue()

def convertImage(img):
    image_file = img.convert('1') # convert image to black and white
    return image_file

def writeimage(pil_img, output_path):
    global counter
    image_path = os.path.join(output_path, f"{counter}.png")
    pil_img.save(image_path)
    print(f"save {image_path} successfull")
    counter+=1
    
def getImage():
    image_folder = 'Screenshots'
    loader = FrameLoader(image_folder)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(convertImage, (image)) for image in loader]
        
        for res in as_completed(futures):
            icp.put(res.result())

def main():
    getImage()
    print(icp.__len__())
    output_path = 'results'
    with ThreadPoolExecutor(max_workers=5) as excutor:
        futures = [excutor.submit(writeimage, (icp.get(), output_path)) for _ in range(icp.__len__())]



    
if __name__ == "__main__":
    t1 = timeit.default_timer()
    main()
    print(f"total time taken to complete is {timeit.default_timer()-t1}")