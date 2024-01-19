from dataclasses import dataclass, field
import random
from typing import List, Dict
from statistics import mean
from config import read_config
import collections
conf = read_config()
TIME_OUT = 1000000

@dataclass
class EqSetInfo:
    """
    first OVERALL latency of leon
    current OVERALL latency of leon 
    how leon opt this query?
    what query?
    """
    first_latency: float = TIME_OUT
    current_latency: float = TIME_OUT
    opt_time: float = -TIME_OUT
    query_ids: List[str] = field(default_factory=list)
    query_dict: Dict[str, float] = field(default_factory=dict)
    eqset_latency: float = TIME_OUT


class SubplanCost(
        collections.namedtuple(
            'SubplanCost',
            ['subplan', 'cost'],
        )):
    """A collected training data point; wrapper around (subplan, goal, cost).

    Attributes:

      subplan: a balsa.Node.
      goal: a balsa.Node. (deprecated in LEON version)
      cost: the cost of 'goal'.  Specifically: start from subplan, eventually
        reaching 'goal' (joining all leaf nodes & with all filters taken into
        account), what's the cost of the terminal plan?
    """

    # Unused fields: goal
    def ToSubplanGoalHint(self, with_physical_hints=False):
        """subplan's hint_str()--optionally with physical ops--and the goal."""
        return 'subplan=\'{}\''.format(
            self.subplan.hint_str(with_physical_hints),
            ','.join(sorted(self.goal.leaf_ids(alias_only=True))))

    # Unused fields: goal
    def __repr__(self):
        """Basic string representation for quick inspection."""
        return 'SubplanGoalCost(subplan=\'{}\', goal=\'{}\', cost={})'.format(
            self.subplan.hint_str(),
            ','.join(sorted(self.subplan.leaf_ids(alias_only=True))), self.cost)


class Experience:
    def __init__(self, eq_set) -> None:
        # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids]] 
        self.MinEqNum = int(conf['leon']['MinEqNum'])
        self.MaxEqSets = int(conf['leon']['MaxEqSets'])
        self.LargeTimout = TIME_OUT
        self.__exp = dict() # k is the join_ids of an eq (str), v is experience of an eq (list)
        self.__eqSet = dict() # save limited # of eq. Some eqs are in __exp, but not in __eqSet
        # self.__eqsetTime = dict() # 每个等价类
        for i in eq_set:

            # todo: hand crafted tuned
            # self.__eqSet[i] = 2000
            # self.__exp[i] = []
            self.AddEqSet(i)
            self.GetEqSet()[i].eqset_latency = 2 * TIME_OUT
            self.AppendExp(i, [])
    
    def OnlyGetExp(self):
        return self.__exp
        

    def GetQueryId(self, eq: EqSetInfo):
        return self.GetEqSet()[eq].query_ids

    def GetEqSetKeys(self):
        return self.__eqSet.keys()
    
    def GetExpKeys(self):
        return self.__exp.keys()

    def AppendExp(self, eq, plan: list):
       #     print(plan)
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        if self.haveEq(eq):
            self.GetExp(eq).append(plan)
        else:
            if plan:
                self.__exp[eq] = [plan]
            else:
                self.__exp[eq] = []
    
    def haveEq(self, eq):
        return self.GetExp(eq)

    def IsJoinIdsValid(self, join_ids):
        if join_ids in self.GetEqSet().keys():
            if self.GetEqSet()[join_ids] != self.LargeTimout:
                return True
        return False

    def isCache(self, eq, plan):
        if self.haveEq(eq):
            for curr_plan in self.GetExp(eq):
                if curr_plan[0].info['sql_str'] == plan[0].info['sql_str'] and curr_plan[0].hint_str() == plan[0].hint_str(): # sql 和 hint 都相同
                    return True
        return False
    
    def ChangeTime(self, eq, plan):
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        for i, curr_plan in enumerate(self.__exp[eq]):
            if curr_plan[0].info['sql_str'] == plan[0].info['sql_str'] and curr_plan[0].hint_str() == plan[0].hint_str():
                self.__exp[eq][i][0].info['latency'] = curr_plan[0].info['latency']
        

    def GetExp(self, eq) -> list:
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        return self.__exp.get(eq)

    def GetEqSet(self) -> dict:
        return self.__eqSet

    def _getEqNum(self):
        # return len(self.GetEqSet())
        return sum(value.opt_time != 0 for value in self.GetEqSet().values())

    def GetPlanNum(self):
        num = 0
        for eq in self.GetEqSetKeys():
            num += len(self.GetExp(eq))
        return num

    def _collectTime(self):
        for eq in self.GetEqSetKeys():
            if self.GetEqSet()[eq].eqset_latency == 2 * TIME_OUT:
                continue
            average = 0
            cnt = 0
            if len(self.GetExp(eq)) > 0:
                for plan in self.GetExp(eq):
                    if plan[0].info['latency'] != TIME_OUT:
                        cnt += 1
                        average += plan[0].info['latency']
                if cnt == 0:
                    self.GetEqSet()[eq].eqset_latency = TIME_OUT
                else:
                    self.GetEqSet()[eq].eqset_latency = average / cnt

    def collectRate(self, eq, first_time, tf, query_id):
        """
        eq: 等价类
        first_time: 第一次leon执行时间
        tf: 当前leon在最终计划中使用该Eq的执行时间
        query_id: 区分查询语句
        """
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        """ Calculate Average Time to rank Eqs.  
        if eq in self.__eqsetTime:
            if query_id in self.__eqsetTime[eq]:
                opt_time = self.__eqsetTime[eq][query_id]
                self.__eqsetTime[eq][query_id] = max(first_time - tf, opt_time)
            else:
                self.__eqsetTime[eq] = dict()
        """

        if eq in self.GetEqSetKeys():
            # rate_old = self.GetEqSet()[eq][2]
            # if self.GetEqSet()[eq].opt_time != 0: # 仅仅新增等价类的重要性
            #     return
            opt_time = first_time - tf
            query_ids = self.GetEqSet()[eq].query_ids
            query_dict = self.GetEqSet()[eq].query_dict
            eqset_latency = self.GetEqSet()[eq].eqset_latency
            if query_id not in query_ids:
                query_ids.append(query_id)
            if query_id not in query_dict.keys():
                query_dict[query_id] = opt_time
            else:
                query_dict[query_id] = max(opt_time, query_dict[query_id])
            
            self.GetEqSet()[eq] = EqSetInfo(first_latency=first_time,
                                            current_latency=tf,
                                            opt_time=mean(query_dict.values()),
                                            query_ids=query_ids,
                                            query_dict=query_dict,
                                            eqset_latency=eqset_latency)

    def DeleteEqSet(self, sql_id):
        self._collectTime()
        EqNum = self._getEqNum()
        if EqNum < self.MinEqNum:
            return
        allSet = list(self.GetEqSet().items())
        # list of (key, value)
        # allSet.sort(key=lambda x: (x[1].opt_time, len(x[0])), reverse=True) # 优先删小的等价类
        allSet.sort(key=lambda x: (x[1].eqset_latency, len(x[0])), reverse=True) # 优先删小的等价类
        def remove_matching_sets(all_set, sql_id):
            """
            Remove sets from all_set if their query_ids contain common elements with sql_id.
            """
            updated_set = [item for item in all_set if not (any(x in sql_id for x in item[1].query_ids) or item[1].eqset_latency == TIME_OUT)]
            return updated_set
        allSet = remove_matching_sets(allSet, sql_id)
        deletenum = min(int(EqNum * 0.15), len(allSet))
        if EqNum - deletenum < self.MinEqNum:
            return
        for i in range(deletenum):
            k, _ = allSet[len(allSet) - 1 - i]
            self.GetEqSet().pop(k)

    def DeleteOneEqset(self, eq):
        del self.GetEqSet()[eq]

    def AddEqSet(self, eq, query_id=None):
        if self._getEqNum() < self.MaxEqSets: # Limit the Total Number of EqSet
            temp = eq.split(',') # sort
            eq = ','.join(sorted(temp))
            if eq not in self.GetEqSetKeys():
                self.GetEqSet()[eq] = EqSetInfo()
                if query_id:
                    self.GetEqSet()[eq].query_ids.append(query_id)
                if not self.GetExp(eq):
                    self.__exp[eq] = []
            else:
                if query_id is not None:
                    if query_id not in self.GetEqSet()[eq].query_ids:
                        self.GetEqSet()[eq].query_ids.append(query_id)
    

    def Getpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.__eqSet.keys():
            # if len(self.GetExp(eq)) < 8: 
            #     continue
            # min_dict = dict()
            # for j in self.GetExp(eq):
            #     if j[0].info['sql_str'] not in min_dict:
            #         min_dict[j[0].info['sql_str']] = j
            #     else:
            #         if j[0].cost < min_dict[j[0].info['sql_str']][0].cost:
            #             min_dict[j[0].info['sql_str']] = j
            for i, j in enumerate(self.GetExp(eq)):
                # for k in min_dict.values():
                for k_index in range(i + 1, len(self.GetExp(eq))): 
                    k = self.GetExp(eq)[k_index]   
                    if (j[0].cost == k[0].cost):
                        continue
                    if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同   
                        continue
                    if (j[0].info['sql_str'] != k[0].info['sql_str']):
                        continue
                    # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                        continue
                    tem = [j, k]
                    # tem.append(j)
                    # tem.append(k)
                    # tem.append(j[1]+j[2])
                    # tem.append(k[1]+k[2])
                    # 初始pairloss
                    pairs.append(tem)
        return pairs

    def PreGetpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.__exp.keys():
            # if len(self.GetExp(eq)) < 8: 
            #     continue
            min_dict = dict()
            for j in self.GetExp(eq):
                if j[0].info['sql_str'] not in min_dict:
                    min_dict[j[0].info['sql_str']] = j
                else:
                    if j[0].cost < min_dict[j[0].info['sql_str']][0].cost:
                        min_dict[j[0].info['sql_str']] = j
            for i, j in enumerate(self.GetExp(eq)):
                # all_elements = list(range(i + 1, len(self.GetExp(eq))))
                # for k_index in random.sample(all_elements, min(10, len(all_elements))):
                # for k_index in range(i + 1, len(self.GetExp(eq))): 
                #     k = self.GetExp(eq)[k_index]   
                for k in min_dict.values():
                    

                    if (j[0].info['sql_str'] != k[0].info['sql_str']):
                        continue
                    # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                        continue
                    tem = [j, k]
                    # tem.append(j)
                    # tem.append(k)
                    # tem.append(j[1]+j[2])
                    # tem.append(k[1]+k[2])
                    # 初始pairloss
                    pairs.append(tem)
        return pairs

