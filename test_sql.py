import re
from util import envs
from config import read_config

conf = read_config()
train_files, training_query = envs.load_train_files(conf['leon']['workload_type'])

a = []
for sql_query in training_query:
    # 提取FROM和WHERE之间的内容
    from_where_content = re.search('FROM(.*)WHERE', sql_query.replace("\n", "")).group(1)

    # 提取别名
    aliases = re.findall(r'AS (\w+)', from_where_content)
    
    aliases = ",".join(sorted(aliases))
    a.append(aliases)

print(len(set(a)))