# import time
import ray
ray.init()

@ray.remote
def f(x):
    # time.sleep(1)
    return x * x

@ray.remote
def remote_hi():
    import os
    import socket
    return f"Running on {socket.gethostname()} in pid {os.getpid()}"

future = remote_hi.remote()
print(ray.get(future))

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures)) # [0, 1, 4, 9]