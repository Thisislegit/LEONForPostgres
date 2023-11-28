import ray
import pickle
import json

from ray import cluster_utils
from ray_test import FileWriter

address='121.48.161.203:44116'
context = ray.init(address, namespace='ray_namespace')

actor = ray.get_actor('foo')
# actor = FileWriter.options(name="foo", lifetime="detached", get_if_exists=True).remote("file.txt")
print('New task submitted, waiting for completion...')
print(ray.get(actor.complete_all_tasks.remote()))
