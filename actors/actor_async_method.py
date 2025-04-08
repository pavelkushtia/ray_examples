import ray
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize Ray
ray.init()

# Define an Actor
@ray.remote
class Worker:
    def __init__(self):
        self.pending_tasks = 0
        self.executor = ThreadPoolExecutor(max_workers=1)  # Only allow one task at a time

    async def work(self, task):
        # Use a thread pool to execute work synchronously but non-blocking for async methods
        self.pending_tasks += 1
        print(f"Starting task: {task}")
        # time.sleep(5)
        await asyncio.get_event_loop().run_in_executor(self.executor, self._do_work, task)
        self.pending_tasks -= 1
        print(f"Finished task: {task}")
        return f"Returned from : {task}"

    def _do_work(self, task):
        # Simulate blocking work (2 seconds)
        time.sleep(2)

    async def get_pending_tasks(self):
        # This method will return immediately, even during work execution
        return self.pending_tasks

# Create the worker actor
worker = Worker.remote()

# Asynchronously assign tasks and check pending tasks
async def main():
    # Dispatch multiple tasks but only one `work` will execute at a time
    result_ids = [
        worker.work.remote("Task 1"),
        worker.work.remote("Task 2"),
        worker.work.remote("Task 3")
    ]
    # Check pending tasks while work is running
    completed, pending = ray.wait(result_ids)
    print(ray.get(completed))

    pending_tasks = await worker.get_pending_tasks.remote()
    while pending_tasks > 0:
        pending_tasks = await worker.get_pending_tasks.remote()
        print(f"Pending tasks: {pending_tasks}")
        await asyncio.sleep(2)
        completed, pending = ray.wait(result_ids)
        print(ray.get(completed))
        print(f"number of completed = {len(completed)}") # This number does not get updated .. aleays 1 why..? does it show only how many got updated after the last get. 
        print(f"number of pending = {len(pending)}") # Same as this , this is always 2 .. but why? 


    completed, pending = ray.wait(result_ids)
    print(ray.get(completed))
    print(f"number of completed = {len(completed)}") # This number does not get updated .. aleays 1 why..? does it show only how many got updated after the last get. 
    print(f"number of pending = {len(pending)}") # Same as this , this is always 2 .. but why? 
# Run the async event loop
asyncio.run(main())

# Shutdown Ray
ray.shutdown()