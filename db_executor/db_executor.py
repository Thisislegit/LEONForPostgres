import asyncio
import asyncpg


class Worker:
    def __init__(self, port, host, user, password, database, pool=None):
        self.port = port
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.pool = pool
        self.available = True

    async def initialize(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(user=self.user, password=self.password, 
                                                  database=self.database, host=self.host, 
                                                  port=self.port, max_size=1)
        self.available = True

    async def query(self, query):
        self.available = False
        async with self.pool.acquire() as con:
            result = await con.execute(query)
        self.available = True
        return result

class QueryScheduler_v1:
    def __init__(self, workers):
        self.workers = workers

    async def execute(self, queries):
        results = []
        for query in queries:
            worker = await self.find_available_worker()
            result = await worker.query(query)
            results.append(result)
        return results

    async def find_available_worker(self):
        while True:
            for worker in self.workers:
                if worker.available:
                    return worker
            await asyncio.sleep(0.1)  

class QueryScheduler_v2:
    def __init__(self, workers):
        self.workers = workers
        self.worker_queue = asyncio.Queue()

    async def schedule_queries(self, queries):
        results = await asyncio.gather(*(self.schedule_query(query) for query in queries))
        return results

    async def schedule_query(self, query):
        worker = await self.get_worker()
        result = await worker.query(query)
        await self.release_worker(worker)
        return result

    async def get_worker(self):
        if self.worker_queue.empty():
            for worker in self.workers:
                await self.worker_queue.put(worker)
        return await self.worker_queue.get()

    async def release_worker(self, worker):
        await self.worker_queue.put(worker)

# Usage
async def main():
    workers = [Worker(port=5432, host='localhost', user='user', password='password', database='db') for _ in range(5)]
    # for worker in workers:
    #     await worker.initialize()
    await asyncio.gather(*(worker.initialize() for worker in workers))

    scheduler = QueryScheduler_v2(workers)

    queries = ["SELECT * FROM table1", "SELECT * FROM table2"]  # Add your queries
    # results = await scheduler.execute(queries)
    results = await scheduler.schedule_queries(queries)
    print(results)

asyncio.run(main())