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
        self.send_task = 0

    def write_file(self):
        with open(self.file_path, 'ab') as file:
            time.sleep(2)
            file.write(b'hello world\n')
        self.completed_tasks += 1
        print(1111111)
    
    def Add_task(self):
        self.send_task += 1
    
    def complete_all_tasks(self):
        print("Complete:", self.completed_tasks)
        if self.completed_tasks == self.send_task:
            return True
        else:
            return False

if __name__ == "__main__":
    # address='121.48.161.203:7777'
    context = ray.init(namespace='ray_namespace')
    print(context.address_info)
    file_path = "./file.txt"
    writer = FileWriter.options(name="foo", lifetime="detached").remote(file_path)
    for _ in range(1):
        ray.get(writer.Add_task.remote())
        object_id = writer.write_file.remote()
    # keep this program alive
    while True:
        time.sleep(10)


    