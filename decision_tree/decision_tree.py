from numpy import argmax
import random
from collections import Counter
import math
import uuid

class DecisionTree():
    uid = 0
    is_trained = False
    criteria = None
    nb_feature = None
    feature_types = None
    feature_index = None
    feature_type = None
    threshold = None
    parent = None
    subtrees = None
    subtree_map = None
    value = None
    verbose = None
    
    def __init__(self, parent=None, verbose=False):
        self.parent = parent
        self.id = DecisionTree.get_uid()
        self.verbose = verbose
    
    def train(self, data, criteria):
        print('generating tree node')
        assert len(data) > 0 and len(data[0]) > 1
        self.is_trained = True
        self.criteria = criteria
        self.nb_feature = len(data[0]) - 1
        self.feature_types = []
        for i in range(self.nb_feature):
            if type(data[0][i]) is str:
                self.feature_types.append('nominal')
            elif type(data[0][i]) is int or type(data[0][i]) is float:
                self.feature_types.append('numerical')
        self.feature_index, cutpoints = self.select_feature(data, self.criteria)        
        if len(cutpoints) > 1:
            data = sorted(data, key = lambda x:x[self.feature_index])
            if self.verbose: print(f'branch node: {(self.parent.id if self.parent else None)} --> {self.id}')
            self.subtrees = []
            if self.feature_types[self.feature_index] == 'numerical':
                self.feature_type = 'numerical'
                self.threshold = data[cutpoints[1]][self.feature_index]
                if self.verbose: print(f'using numerical feature {self.feature_index}, threshold: {self.threshold}\n')
                self.subtrees.append(DecisionTree(self, verbose=self.verbose).train(data[cutpoints[0]:cutpoints[1]], criteria))
                self.subtrees.append(DecisionTree(self, verbose=self.verbose).train(data[cutpoints[1]:], criteria))
            elif self.feature_types[self.feature_index] == 'nominal':
                self.feature_type = 'nominal'
                self.subtree_map = dict()
                if self.verbose: print(f'using nominal feature {self.feature_index}\n')
                for i in range(len(cutpoints)):
                    subtree = None
                    if i == len(cutpoints)-1:
                        subtree = DecisionTree(self, verbose=self.verbose).train(data[cutpoints[i]:], criteria)
                    else:
                        subtree = DecisionTree(self, verbose=self.verbose).train(data[cutpoints[i]:cutpoints[i+1]], criteria)
                    self.subtrees.append(subtree)
                    self.subtree_map[data[cutpoints[i]][self.feature_index]] = subtree
        else:
            labels = (x[-1] for x in data)
            cl = Counter(labels)
            self.value = cl.most_common(1)[0][0]       
            if self.verbose: print(f'leaf node: {self.value}; {(self.parent.id if self.parent else None)} --> {self.id}\n')
        return self
    
    def infer(self, data):
        assert self.is_trained
        assert len(data)==self.nb_feature
        if self.subtrees is None:
            return self.value
        
        if self.feature_type == 'numerical':
            if data[self.feature_index] < self.threshold:
                return self.subtrees[0].infer(data)
            else:
                return self.subtrees[1].infer(data)
        elif self.feature_type == 'nominal':
            return self.subtree_map[data[self.feature_index]].infer(data)
    
    def select_feature(self, data, criteria):
        criteria_values = []
        cutpoints = []
        for i in range(self.nb_feature):
            if self.verbose: print(f'testing feature {i}')
            data = sorted(data, key=lambda x: x[i])
            if self.feature_types[i] == 'numerical':
                current_max = 0
                max_cutpoint = 0
                for cutpoint in range(1,len(data)):
                    cv = criteria(data, [0, cutpoint])
                    if cv > current_max:
                        current_max = cv
                        max_cutpoint = cutpoint
                criteria_values.append(current_max)
                cutpoints.append([0, max_cutpoint])
            elif self.feature_types[i] == 'nominal':
                cutpoint = [0]
                for j in range(1, len(data)):
                    if data[j][i] != data[j-1][i]:
                        cutpoint.append(j)
                cv = criteria(data, cutpoint)
                criteria_values.append(cv)
                cutpoints.append(cutpoint)
        
        max_ind = argmax(criteria_values)
        if criteria_values[max_ind] > 0:
            return max_ind, cutpoints[max_ind]
        else:
            return max_ind, [0]
        
    def print_tree(self):
        pass
    
    @classmethod
    def get_uid(cls):
        return uuid.uuid4().hex
                
    @classmethod
    def information_gain(cls, data, cutpoints):
        def entropy(d):
            ent = 0
            c = Counter(d)
            for value in c:
                ent -= c[value]/len(d)*math.log2(c[value]/len(d))
            return ent
        values = [x[-1] for x in data]
        gain = entropy(values)
        for i in range(len(cutpoints)):
            if i == len(cutpoints)-1:
                slice_ = values[cutpoints[i]:]
            else:
                slice_ = values[cutpoints[i]:cutpoints[i+1]]
            gain -= len(slice_)/len(values)*entropy(slice_)
        return gain
    
    @classmethod
    def gain_ratio(cls, data, cutpoints):
        intrinsic_value = 0
        for i in range(len(cutpoints)):
            if i == len(cutpoints)-1:
                intrinsic_value -= (len(data)-cutpoints[i])/len(data)*math.log2((len(data)-cutpoints[i])/len(data))
            else:
                intrinsic_value -= (cutpoints[i+1]-cutpoints[i])/len(data)*math.log2((cutpoints[i+1]-cutpoints[i])/len(data))
        ratio = (cls.information_gain(data, cutpoints)/intrinsic_value if intrinsic_value != 0 else 0)
        return ratio
    
    @classmethod
    def gini_index(cls, data, cutpoints):
        def gini(d):
            gini = 1
            c = Counter(d)
            for value in c:
                gini -= (c[value]/len(d))**2
            return gini
        values = [x[-1] for x in data]
        gini_index = 0
        for i in range(len(cutpoints)):
            if i == len(cutpoints)-1:
                slice_ = values[cutpoints[i]:]
            else:
                slice_ = values[cutpoints[i]:cutpoints[i+1]]
            gini_index += len(slice_)/len(values)*gini(slice_)
#         print(gini_index)
        return 1 - gini_index