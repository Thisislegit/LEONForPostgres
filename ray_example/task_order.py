import time
import ray

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, addition):
        self.value += addition
        return self.value

counter = Counter.remote()

# Submit task from a worker
@ray.remote
def submitter(value):
    return ray.get(counter.add.remote(value))

# Simulate delayed result resolution.
@ray.remote
def delayed_resolution(value):
    time.sleep(5)
    return value

# Submit tasks from different workers, with
# the first submitted task waiting for
# dependency resolution.
value0 = submitter.remote(delayed_resolution.remote(1))
value1 = submitter.remote(2)

# Output: 3. The first submitted task is executed later.
print(ray.get(value0))
# Output: 2. The later submitted task is executed first.
print(ray.get(value1))