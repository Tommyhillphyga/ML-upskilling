
import threading
import time

start_time = time.perf_counter()
def cal_square(numbers):
    print("Calculate square numbers")
    for n in numbers:
        time.sleep(0.2)
        print('Square:', n*n)

def cal_cube(numbers):
    print("Calculate cube numbers")
    for n in numbers:
        time.sleep(0.2)
        print('Cube:', n*n*n)

if __name__ == "__main__":
    arr = [2,3,8,9]

    t1 = threading.Thread(target=cal_square, args=(arr,))
    t2 = threading.Thread(target=cal_cube, args=(arr,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    end_time = time.perf_counter()
    print("Done in ", end_time - start_time)