class Experience:
    def __init__(self, eq_set) -> None:
        # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids]] 
        self._exp = dict() # k is the join_ids of an eq (str), v is experience of an eq (list)
        self._eqSet = dict() # save limited # of eq. Some eqs are in _exp, but not in _eqSet
        for i in eq_set:
            # todo: hand crafted tuned
            self._eqSet[i] = 2000
            self._exp[i] = []
        self.MinEqNum = 9
        self.LargeTimout = 90000

    def GetEqSetKeys(self):
        return self._eqSet.keys()

    def AppendExp(self, eq, plan: list):
       #     print(plan)
        if self.haveEq(eq):
            self.GetExp(eq).append(plan)
        else:
            self._exp[eq] = [plan]
    
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
                if curr_plan[1] == plan[1] and curr_plan[2] == plan[2]: # sql 和 hint 都相同
                    return True
        return False

    def GetExp(self, eq) -> list:
        return self._exp.get(eq)

    def GetEqSet(self) -> dict:
        return self._eqSet

    def _getEqNum(self):
        return len(self.GetEqSet())

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
                    if plan[3] != 90000:
                        cnt += 1
                        average += plan[3]
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
        if len(self.GetEqSetKeys()) < 25: # Limit the Total Number of EqSet
            if eq not in self.GetEqSetKeys():
                self.GetEqSet()[eq] = 90000
                if not self.GetExp(eq):
                    self._exp[eq] = []

    def Getpair(self):
        """
        a train pair
        [[j cost, j latency, j query_vector, j node], [k ...]], ...
        """
        pairs = []
        for eq in self.GetEqSetKeys():
            for j in self.GetExp(eq):
                for k in self.GetExp(eq):
                    if (j[2] == k[2]) and (j[1] == k[1]): # sql 和 hint 都相同
                        continue
                    if (j[3] == k[3]): # latency 相同
                        continue
                    tem = []
                    tem.append([j[0], j[3], j[4][0], j[4][1]])
                    tem.append([k[0], k[3], k[4][0], k[4][1]])
                    # tem.append(j[1]+j[2])
                    # tem.append(k[1]+k[2])
                    # 初始pairloss
                    pairs.append(tem)
        return pairs

