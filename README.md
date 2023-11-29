# LEON System Documentation

This document provides an overview of the LEON system, focusing on its training and inference components as implemented in `LEON_train.py` and `Leon_server.py`.

## Pre-requests
### GUC
Set following guc for enabling (enable_leon is off as default, not_cali is off as default)
```bash
enable_leon = on;
not_cali = off;
```
### Postgres Config
Load ./conf/LEON-postgresql.conf into postgresql

## LEON_train.py - Training Process
### TODO:
- [ ] write nodes file with ray
- [ ] pick_node
- [ ] train model


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
- [x] Fix json load errors (parsed from postgres)
- [x] Ingretation of transformer
- [x] Test inference efficiency and bottleneck
![Effciency](./Figs/efficiency.jpg)
- [x] Test trained model ([SeqTransformer](https://github.com/liang-zibo/DACE))

    **pre-trained model for 10 templates**

    ![seqformer](./Figs/pre_train_model.jpg)
    **pre-trained model for 30 templates (omit 28c)**

    23个优化分别是['1d', '6f', '7a', '9b', '9c', '10a', '10b', '12a', '15c', '15d', '16b', '16c', '17a', '17e', '18a', '18c', '19a', '19d', '23a', '23c', '24b', '25c', '26c'] 
    
    76个劣化pg

    ![seqformer](./Figs/pre_train_model_all.jpg)
    **Warning: Query Regression on 28C**

    ![seqformer](./Figs/28c.jpg)

    **End to End Runtime**

    |        | Runtime       |          |
    | :------: | :------: | ------: |
    ｜          | Postgres | SeqFormer |
    | with 28c      | 190.9s   | 26006.6s  |
    | without 28c   | 190.9s   | 233.7s    |

- [x] Multiple Database Execution for acceleration.
- [x] Add Eqset Judgement for model inference

    **End to End Runtime**

    25个结果优于pg，74个结果劣于pg

    优于pg的查询['2a', '3a', '3c', '4a', '6a', '6b', '6d', '6f', '7c', '8a', '8b', '9c', '10a', '10b', '10c', '12a', '13c', '15c', '16a', '18a', '18b', '20a', '20b', '25c', '26a']


   ![seqformer](./Figs/eqset_pre_train_model_all.jpg)

    |        | Runtime       |        |
    | :------: | :------: | ------: |
    ｜          | Postgres | SeqFormer |
    |       | 181.7s   | 185.8s  |
    
