import ray
from ray.util import ActorPool
import psycopg2
import os

def load_sql(file_list: list):
    """
    :param file_list: list of query file in str
    :return: list of sql query string
    """
    sqls = []
    for file_str in file_list:
        sqlFile = '/data1/wyz/online/LEONForPostgres/join-order-benchmark/' + file_str + '.sql'
        if not os.path.exists(sqlFile):
            raise IOError("File Not Exists!")
        with open(sqlFile, 'r') as f:
            data = f.read().splitlines()
            sql = ' '.join(data)
        sqls.append(sql)
        f.close()
    return sqls

ray.init()
@ray.remote
class ActorThatQueries:
    def __init__(self, port):
        # Initialize and configure your database connection here
        self.db = psycopg2.connect(database="postgres", user="wyz", password="wangyuze", host="localhost", port=port)

    def query_db(self, sql):
        # Implement the logic to query the database
        with self.db.cursor() as cursor:
            cursor.execute(sql)
            res = cursor.fetchall()
        return res

# Create a list of actors
actors = [ActorThatQueries.remote(port) for port in [1120, 1125, 1130]]

# Initialize the actor pool
pool = ActorPool(actors)

files = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a',
             '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', 
             '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', 
             '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', 
             '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c',
             '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a',
             '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b',
             '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', 
             '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
             '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c', 'end']
query_values = load_sql(files)
# Define the range of values to query

# Execute queries in parallel
results = list(pool.map_unordered(lambda a, v: a.query_db.remote(v), query_values))

# Process the results
for result in results:
    # Handle each result
    print(result)