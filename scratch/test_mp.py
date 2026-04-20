import multiprocessing
import os
import time

def worker(q):
    print(f"Worker PID: {os.getpid()}")
    q.put(f"Hello from {os.getpid()}")

if __name__ == "__main__":
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    print("Process started")
    print(q.get())
    p.join()
    print("Done")
