import threading
from concurrent.futures import ThreadPoolExecutor
import time
import random


def myThread(i):
    print("Thread {}: started".format(i))
    time.sleep(random.randint(1,5))
    print("Thread {}: finished".format(threading.current_thread()))


    
def main():
    threads = []
    with ThreadPoolExecutor(max_workers=3) as excutor:
        i = 2
        tasks = excutor.submit(myThread, (i,))
    print("Enumerating: {}".format(threading.enumerate()))

if __name__ == '__main__':
    main()