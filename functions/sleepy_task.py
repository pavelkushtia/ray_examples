import ray
import timeit

def slow_task(x): 
    import time
    time.sleep(2) # Do something sciency/business return x

@ray.remote
def remote_task(x):
    return slow_task(x) 

things = range(10)

very_slow_result = map(slow_task, things)
slowish_result = map(lambda x: remote_task.remote(x), things)

slow_time = timeit.timeit(lambda: list(very_slow_result), number=1)
fast_time = timeit.timeit(lambda: list(ray.get(list(slowish_result))), number=1) 
print(f"In sequence {slow_time}, in parallel {fast_time}")
