import collections
import contextlib
import socket

import psycopg2
import psycopg2.extensions
import ray
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE
from select import select
from . import plans_lib
from . import postgres


# Change these strings so that psycopg2.connect(dsn=dsn_val) works correctly
# for local & remote Postgres.

# JOB/IMDB.
# LOCAL_DSN = "postgres://psycopg:psycopg@localhost/imdb"

from config import read_config
conf = read_config()


database = conf['PostgreSQL']['database']
user = conf['PostgreSQL']['user']
password = conf['PostgreSQL']['password']
host = conf['PostgreSQL']['host']
port = conf['PostgreSQL']['port']
LOCAL_DSN = ""
REMOTE_DSN = ""
leon_port = conf['leon']['Port']

# TPC-H.
# LOCAL_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"
# REMOTE_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"

# A simple class holding an execution result.
#   result: a list, outputs from cursor.fetchall().  E.g., the textual outputs
#     from EXPLAIN ANALYZE.
#   has_timeout: bool, set to True iff an execution has run outside of its
#     allocated timeouts; False otherwise (e.g., for non-execution statements
#     such as EXPLAIN).
#   server_ip: str, the private IP address of the Postgres server that
#     generated this Result.
Result = collections.namedtuple(
    'Result',
    ['result', 'has_timeout', 'server_ip'],
)


# ----------------------------------------
#     Psycopg setup
# ----------------------------------------


def wait_select_inter(conn):
    while 1:
        try:
            state = conn.poll()
            if state == POLL_OK:
                break
            elif state == POLL_READ:
                select([conn.fileno()], [], [])
            elif state == POLL_WRITE:
                select([], [conn.fileno()], [])
            else:
                raise conn.OperationalError("bad state from poll: %s" % state)
        except KeyboardInterrupt:
            conn.cancel()
            # the loop will be broken by a server error
            continue


psycopg2.extensions.set_wait_callback(wait_select_inter)


@contextlib.contextmanager
def Cursor():
    """Get a cursor to local Postgres database."""
    # TODO: create the cursor once per worker node.
    conn = psycopg2.connect(database=database, user=user,
                            password=password, host=host, port=port)
    conn.set_client_encoding('UTF8')

    conn.set_session(autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET client_encoding TO 'UTF8';")
            cursor.execute(f"set leon_port={leon_port};")
            cursor.execute("load 'pg_hint_plan';")

            yield cursor
    finally:
        conn.close()


# ----------------------------------------
#     Postgres execution
# ----------------------------------------


def _SetGeneticOptimizer(flag, cursor):
    # NOTE: DISCARD would erase settings specified via SET commands.  Make sure
    # no DISCARD ALL is called unexpectedly.
    assert cursor is not None
    assert flag in ['on', 'off', 'default'], flag
    cursor.execute('set geqo = {};'.format(flag))
    assert cursor.statusmessage == 'SET'

def ExecuteRemote(sql, verbose=False, geqo_off=False, timeout_ms=None):
    return _ExecuteRemoteImpl.remote(sql, verbose, geqo_off, timeout_ms)


@ray.remote(resources={'pg': 1})
def _ExecuteRemoteImpl(sql, verbose, geqo_off, timeout_ms):
    with Cursor(dsn=REMOTE_DSN) as cursor:
        return Execute(sql, verbose, geqo_off, timeout_ms, cursor)


def Execute(sql, verbose=False, geqo_off=False, timeout_ms=None, cursor=None):
    """Executes a sql statement.

    Returns:
      A pg_executor.Result.
    """
    # if verbose:
    #  print(sql)

    _SetGeneticOptimizer('off' if geqo_off else 'on', cursor)
    if timeout_ms is not None:
        cursor.execute('SET statement_timeout to {}'.format(int(timeout_ms)))
    else:
        # Passing None / setting to 0 means disabling timeout.
        cursor.execute('SET statement_timeout to 0')
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        has_timeout = False
    except Exception as e:
        if isinstance(e, psycopg2.errors.QueryCanceled):
            assert 'canceling statement due to statement timeout' \
                   in str(e).strip(), e
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.errors.InternalError_):
            print(
                'psycopg2.errors.InternalError_, treating as a' \
                ' timeout'
            )
            print(e)
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.OperationalError):
            if 'SSL SYSCALL error: EOF detected' in str(e).strip():
                # This usually indicates an expensive query, putting the server
                # into recovery mode.  'cursor' may get closed too.
                print('Treating as a timeout:', e)
                result = []
                has_timeout = True
            else:
                # E.g., psycopg2.OperationalError: FATAL: the database system
                # is in recovery mode
                raise e
        else:
            raise e
    try:
        pass
        # _SetGeneticOptimizer('default', cursor)
    except psycopg2.InterfaceError as e:
        # This could happen if the server is in recovery, due to some expensive
        # queries just crashing the server (see the above exceptions).
        assert 'cursor already closed' in str(e), e
        pass
    ip = socket.gethostbyname(socket.gethostname())
    return Result(result, has_timeout, ip)



@contextlib.contextmanager
def MyCursor(database_port):
    """Get a cursor to local Postgres database."""
    # TODO: create the cursor once per worker node.
    conn = psycopg2.connect(database=database, user=user,
                            password=password, host=host, port=database_port)
    conn.set_client_encoding('UTF8')
    conn.set_session(autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET client_encoding TO 'UTF8';")
            cursor.execute("load 'pg_hint_plan';")

            yield cursor
    finally:
        conn.close()

def actor_call(actor, item):
    return actor.ActorExecute.remote(item)

def actor_call_leon(actor, item):
    return actor.ActorExecute_leon.remote(item)

@ray.remote
class ActorThatQueries:
    def __init__(self, actor_port, our_port):
        # Initialize and configure your database connection here
        self.port = actor_port
        self.our_port = our_port
        self.TIME_OUT = 1000000

    def ActorExecute(self, plan):
        # Implement the logic to query the database
        exp = plan[0]
        node = plan[0][0]
        timeout = plan[1]
        hint_node = plans_lib.FilterScansOrJoins(node.Copy())
        explain_str = 'explain(verbose, format json, analyze)'
        comment = hint_node.hint_str()
        sql = hint_node.info['sql_str']

        end_of_comment_idx = sql.find('*/')
        if end_of_comment_idx == -1:
            existing_comment = None
        else:
            split_idx = end_of_comment_idx + len('*/\n')
            existing_comment = sql[:split_idx]
            sql = sql[split_idx:]

        # Fuse hint comments.
        if comment:
            assert comment.startswith('/*+') and comment.endswith('*/'), (
                'Don\'t know what to do with these', sql, existing_comment, comment)
            if existing_comment is None:
                fused_comment = comment
            else:
                comment_body = comment[len('/*+ '):-len(' */')].rstrip()
                existing_comment_body_and_tail = existing_comment[len('/*+'):]
                fused_comment = '/*+\n' + comment_body + '\n' + existing_comment_body_and_tail
        else:
            fused_comment = existing_comment

        if fused_comment:
            s = fused_comment + '\n' + str(explain_str).rstrip() + '\n' + sql
        else:
            s = str(explain_str).rstrip() + '\n' + sql
        with MyCursor(self.port) as cursor:
            result = Execute(s, True, True, timeout, cursor).result
        if not result:
            exp[0].info['latency'] = self.TIME_OUT
        else:
            json_dict = result[0][0][0]
            latency = float(json_dict['Execution Time'])
            exp[0].info['latency'] = latency
        return (exp, plan[2])
    

    def ActorExecute_leon(self, plan):
        # Implement the logic to query the database
        exp = plan[0]
        node = plan[0][0]
        timeout = plan[1]
        explain_str = 'explain(verbose, format json, analyze)'
        sql = node.info['sql_str']

        s = str(explain_str).rstrip() + '\n' + sql

        with MyCursor(self.port) as cursor:
            cursor.execute('SET enable_leon=on;')
            cursor.execute(f"set leon_port={self.our_port};")
            cursor.execute(f"SET leon_query_name='picknode:{plan[3]}';") # 第0个plan 0 
            result = Execute(s, True, True, timeout, cursor).result
        if not result:
            exp[0].info['latency'] = self.TIME_OUT
        else:
            json_dict = result[0][0][0]
            latency = float(json_dict['Execution Time'])
            exp[0].info['latency'] = latency
            if json_dict['Plan']['Total Cost'] != round(plan[3] * 100) / 100:
                print(json_dict['Plan']['Total Cost'], plan[3])
                print(sql)
                return None

        return (exp, plan[2])