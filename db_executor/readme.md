# Async PostgreSQL Query Executor

## Overview

This Python script demonstrates an asynchronous approach to executing SQL queries on a PostgreSQL database using the `asyncpg` library. It features a `Worker` class for handling database connections and two versions of a `QueryScheduler` class (`QueryScheduler_v1` and `QueryScheduler_v2`) for scheduling and executing queries. The implementation leverages Python's `asyncio` coroutines for efficient, non-blocking database operations.

## Features

1. **Worker Class**: Manages individual database connections.
   - Initializes a connection pool with `asyncpg`.
   - Executes queries asynchronously.
   - Maintains availability status for scheduling.

2. **QueryScheduler_v1**:
   - Sequentially finds an available worker and assigns queries.
   - Ensures that each worker is executing one query at a time.

3. **QueryScheduler_v2**:
   - Utilizes an `asyncio.Queue` for managing workers.
   - Schedules and executes queries concurrently.
   - Optimizes the scheduling process, reducing wait times.

## Usage

1. Initialize the `Worker` instances.
2. Choose the scheduler version.
3. Provide a list of SQL queries.
4. Run the script to execute the queries asynchronously.

## Requirements

- Python 3.7+
- `asyncpg` library

## Example

```python
async def main():
    workers = [Worker(port=5432, host='localhost', user='user', password='password', database='db') for _ in range(5)]
    await asyncio.gather(*(worker.initialize() for worker in workers))

    scheduler = QueryScheduler_v2(workers)
    queries = ["SELECT * FROM table1", "SELECT * FROM table2"]  # Add your queries
    results = await scheduler.schedule_queries(queries)
    print(results)

asyncio.run(main())
