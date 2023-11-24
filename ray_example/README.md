# Using Ray Actors Across Different Sessions

This guide demonstrates how to use a Ray actor to store and load datasets across different sessions using the provided Python script.

```python
import ray
import argparse

@ray.remote
class DatasetStore:
    def __init__(self):
        self.ds_store = {}

    def store(self, name, dataset):
        self.ds_store[name] = dataset

    def load(self, name):
        return self.ds_store.get(name)


ray.init(address="auto", namespace="myspace")
ds_store = DatasetStore.options(name="my_store", lifetime="detached", get_if_exists=True).remote()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    if args.save:
        ds = ray.data.range(10)
        ray.get(ds_store.store.remote('ds1', ds))

    if args.load:
        ds1 = ray.get(ds_store.load.remote('ds1'))
        print(ds1.take(10))
```

## Overview

The `DatasetStore` actor is a Ray remote class that serves as an in-memory store for datasets. It can be used across different Ray sessions due to its `detached` lifetime. This means that once created, the actor will remain alive independently of the session that created it, until it is explicitly killed or the Ray cluster is shut down.

## Prerequisites

- Python 3.6 or higher
- [Ray](https://ray.io) installed in your environment (`pip install ray`)
- A running Ray cluster or local Ray instance

## Usage

First, ensure that your Ray cluster is running and accessible. You can start a local Ray instance using the `ray start` command.

### Storing a Dataset

To store a dataset, use the `--save` flag when running the script:

```bash
python your_script.py --save
```

To load a dataset, use the --load flag:

```bash
python your_script.py --load
```



Here is a brief explanation of the script components:

* DatasetStore: A Ray actor class that holds a dictionary to store and retrieve datasets.

* ray.init(address="auto", namespace="myspace"): Initializes connection to the Ray cluster and sets a namespace for the actors.

* DatasetStore.options(...): Sets the actor to be detached, which allows it to persist across sessions.

* --save and --load: Command-line flags to trigger dataset storage or retrieval.

# Using Ray for Efficient Database Execution

This guide explains how to leverage the Ray framework for efficient database querying. Ray allows for parallel execution and easy scaling, which is particularly useful when dealing with multiple database queries. We'll be using Ray actors to create a pool of database connections and execute queries in parallel.

## Prerequisites

- Python 3.x
- Ray (`pip install ray`)
- A database and corresponding Python library (e.g., `psycopg2` for PostgreSQL, `pymysql` for MySQL)

## Setup

First, import the necessary modules and initialize Ray:

```python
import ray
from ray.util import ActorPool

ray.init()
@ray.remote
class ActorThatQueries:
    def __init__(self):
        # Initialize and configure your database connection here
        # Example: self.db = psycopg2.connect(...)

    def query_db(self, val):
        # Implement the logic to query the database
        # Example: res = self.db.execute("SELECT * FROM table WHERE id = %s", (val,))
        # return res

# Create a list of actors
actors = [ActorThatQueries.remote() for _ in range(5)]

# Initialize the actor pool
pool = ActorPool(actors)

# Define the range of values to query
query_values = list(range(100))

# Execute queries in parallel
results = list(pool.map_unordered(lambda a, v: a.query_db.remote(v), query_values))

# Process the results
for result in results:
    # Handle each result
    pass
```


Replace the placeholders and example code with specifics relevant to your database and requirements.





## Reference
[Ray Discussion](https://discuss.ray.io/t/is-it-possible-to-share-objects-between-different-driver-processes/6888)

[Github Issue](https://github.com/ray-project/ray/issues/12635)

[Actor Task Order](https://docs.ray.io/en/latest/ray-core/actors/task-orders.html)