import ray
import threading
import time
import pickle
import json

@ray.remote
class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed_tasks = 0

    def write_file(self):
        with open(self.file_path, 'ab') as file:
            time.sleep(10)
            file.write(b'hello world\n')
        self.completed_tasks += 1
        print(1111111)
    
    def complete_all_tasks(self, task_num):
        print(self.completed_tasks)
        if self.completed_tasks == task_num:
            return True
        else:
            return False

if __name__ == "__main__":
    address='121.48.161.203:7777'
    context = ray.init(address, namespace='ray_namespace')
    print(context.address_info)
    file_path = "./file.txt"
    writer = FileWriter.options(name="foo", lifetime="detached").remote(file_path)
    for _ in range(10):
        object_id = writer.write_file.remote()
    # keep this program alive
    while True:
        time.sleep(10)


    