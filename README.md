# LEON System Documentation

This document provides an overview of the LEON system, focusing on its training and inference components as implemented in `LEON_train.py` and `Leon_server.py`.

## LEON_train.py - Training Process

The training process is broken down into several steps, involving chunking queries, executing them, and using the feedback for training a model.


Queries are processed in batches for efficiency:

```python
chunks = chunk(Querylist, 5)

for chunk in chunks:
    for query in chunk:
        Feedback_1, Nodes = pg.execute(query, leon=on)  # Inference phase

    # Execution phase
    nodes_to_execute = Pick_node(Nodes)
    Feedback_2 = pg.execute(nodes_to_execute.to_sql(),
                            nodes_to_execute.to_hint(), 
                            leon=off) 
    experience = Exp(Feedback_1, Feedback_2)

    # Training phase
    experience.getpair()
    model.train()
```

## Leon_server.py - Inference Process
### TODO:
- [ ] Fix json load errors (parsed from postgres)
- [ ] Ingretation of transformer
- [ ] Test inference efficiency and bottleneck


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


