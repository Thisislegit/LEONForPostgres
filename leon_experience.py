class Experience:
    def __init__(self, eq_set) -> None:
        # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids]] 
        self.__exp = dict() # k is the join_ids of an eq (str), v is experience of an eq (list)
        self.__eqSet = dict() # save limited # of eq. Some eqs are in __exp, but not in __eqSet
        for i in eq_set:

            # todo: hand crafted tuned
            # self.__eqSet[i] = 2000
            # self.__exp[i] = []
            self.AddEqSet(i)
            self.AppendExp(i, [])
        self.MinEqNum = 9
        self.LargeTimout = 90000

    def GetEqSetKeys(self):
        return self.__eqSet.keys()

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

    def GetExp(self, eq) -> list:
        temp = eq.split(',') # sort
        eq = ','.join(sorted(temp))
        return self.__exp.get(eq)

    def GetEqSet(self) -> dict:
        return self.__eqSet

    def _getEqNum(self):
        return sum(value < 90000 for value in self.GetEqSet().values())

    def GetPlanNum(self):
        num = 0
        for eq in self.GetEqSetKeys():
            num += len(self.GetExp(eq))
        return num

    def _collectTime(self):
        for eq in self.GetEqSetKeys():
            average = 0
            cnt = 0
            if len(self.GetExp(eq)) > 0:
                for plan in self.GetExp(eq):
                    if plan[0].info['latency'] != 90000:
                        cnt += 1
                        average += plan[0].info['latency']
                if cnt == 0:
                    self.GetEqSet()[eq] = 90000
                else:
                    self.GetEqSet()[eq] = average / cnt

    def DeleteEqSet(self):
        self._collectTime()
        if self._getEqNum() < self.MinEqNum:
            return
        allSet = list(self.GetEqSet().items())
        allSet.sort(key=lambda x: x[1], reverse=False)
        deletenum = int(self._getEqNum() * 0.3)
        if self._getEqNum() - deletenum < self.MinEqNum:
            return
        for i in range(deletenum):
            k, v = allSet[i]
            if v < 5000:
                self.GetEqSet().pop(k)

    def AddEqSet(self, eq):
        if self._getEqNum() < 25: # Limit the Total Number of EqSet
            temp = eq.split(',') # sort
            eq = ','.join(sorted(temp))
            if eq not in self.GetEqSetKeys():
                self.GetEqSet()[eq] = 90000
                if not self.GetExp(eq):
                    self.__exp[eq] = []

    def Getpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.__eqSet.keys():
            # if len(self.GetExp(eq)) < 8: 
            #     continue
            for j in self.GetExp(eq):
                for k in self.GetExp(eq):
                    if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                        continue
                    if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    # if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.2:
                        continue
                    tem = [j, k]
                    # tem.append(j)
                    # tem.append(k)
                    # tem.append(j[1]+j[2])
                    # tem.append(k[1]+k[2])
                    # 初始pairloss
                    pairs.append(tem)
        return pairs


