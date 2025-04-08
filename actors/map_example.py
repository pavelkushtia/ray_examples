import ray
import time

# Start Ray
ray.init()

# Define an actor class
@ray.remote
class Worker:
    def compute(self, x):
        # Simulate a time-consuming computation
        time.sleep(100)
        return f"Worker processed {x}"

# Create a pool of actors
workers = [Worker.remote()] * 10
pool = ray.util.ActorPool(workers)

# Define the task we want to map over
inputs = [1, 2, 3, 4, 5, 6, 7, 8]

# Use the map function to distribute the tasks among the workers
# Each input from the list will be processed by one of the workers
results = pool.map(lambda worker, x: worker.compute.remote(x), inputs)

# Collect the results
for result in results:
    print(ray.get(result))
