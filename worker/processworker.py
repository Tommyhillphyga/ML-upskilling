import multiprocessing
import time


start_time = time.perf_counter()
def cal_square(numbers):
    print("Calculate square numbers")
    for n in numbers:
        # time.sleep(0.2)
        print('Square:', n*n)

def cal_cube(numbers):
    print("Calculate cube numbers")
    for n in numbers:
        # time.sleep(0.2)
        print('Cube:', n*n*n)


if __name__ == "__main__": 
    arr = [2,3,8,9]

    p1 = multiprocessing.Process(target=cal_square, args=(arr,))
    p2 = multiprocessing.Process(target=cal_cube, args=(arr,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    # cal_square(arr)
    # cal_cube(arr)
    end_time = time.perf_counter()
    print("Done in ", end_time - start_time)