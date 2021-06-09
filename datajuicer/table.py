import random
from datajuicer.utils import *

class Table():
    class Key_Var():
        def __init__(self, key):
            self.key = key
        
        def prepare(self, label_dict, grid):
            return self.key

    class Deviation_Var():
        def __init__(self, default, label):
            self.default = default
            self.label = label

        def deviance(self, data, label_dict):
            ret = []
            for key in self.default:
                if not get(data,key) == self.default[key]:
                    d = f"{label_dict.get(key,key)} = {get(data,key,key)}"
                    ret.append(label_dict.get(d,d))
            if ret == []:
                return "Standard"
            return ", ".join(ret)
        
        def prepare(self, label_dict, grid):
            i = random.randint(1000000000, 9999999999)
            label_dict[f"__{i}__"] = self.label
            for data in grid:
                dev = self.deviance(data, label_dict)
                data[f"__{i}__"] = dev
            return f"__{i}__"

    class Product_Var():
        def __init__(self, keys, label=""):
            self.vars = keys
            for i,key in enumerate(keys):
                if type(key) is str:
                    self.vars[i] = Table.Key_Var(key)

            self.label = label
        

        def prepare(self, label_dict, grid):
            keys = [var.prepare(self.independent_keys, label_dict, grid) for var in self.vars]
                
            i = random.randint(1000000000, 9999999999)
            label_dict[f"__{i}__"] = self.label
            for data in grid:
                val = ", ".join([f"{label_dict[key]} = {get(data,key,key)}" for key in keys])
                data[f"__{i}__"] = val
            return f"__{i}__"

    class Dummy_Var():
        def __init__(self):
            pass

        def prepare(self, label_dict, grid):
            i = random.randint(1000000000, 9999999999)
            for data in grid:
                data[f"__{i}__"] = ""
            return f"__{i}__"

    def __init__(self, grid, independent_keys, dependent_keys, label_dict={}, dim=None, order=None):
        grid = copy.deepcopy(grid)
        self.label_dict = label_dict
        self.independent_keys = [Table.Key_Var(key) if type(key) is str else key for key in independent_keys]
        if not dim is None:
            if len(self.independent_keys)+1 > dim:
                self.independent_keys = [Table.Product_Var(self.independent_keys[0:len(self.independent_keys)-dim])] + self.independent_keys[len(self.independent_keys)-dim:-1]
            elif len(self.independent_keys) +1 < dim:
                self.independent_keys = [Table.Dummy_Var()] * (dim - len(self.independent_keys) -1) + self.independent_keys
        self.independent_keys = [var.prepare(label_dict, grid) for var in self.independent_keys]
        self.dependent_keys = dependent_keys
        self._f = query(grid, self.dependent_keys, group_by=self.independent_keys, return_func=True)

        self.order = {}
        
        for indep_key in self.independent_keys:
            self.order[indep_key] = sorted(list(set([get(data,indep_key,indep_key)  for data in grid])))
        
        if order:
            indep_keys = []
            for key_or_label in order:
                if key_or_label in self.independent_keys:
                    indep_keys.append(key_or_label)
                elif key_or_label in self.label_dict.values():
                    indep_keys.append([key for (key,val) in self.label_dict.items() if val==key_or_label and key in self.independent_keys][0])

            for indep_key in indep_keys:
                prior = self.order[indep_key]
                reverse = {v:[k for k in label_dict if label_dict[k] == v and k in prior]  for v in self.label_dict.values()}
                edited = [[val] if val in prior else (reverse[val] if val in reverse else [None]) for val in order[key_or_label]]
                edited = sum(edited, [])
                edited = [val for val in edited if not val is None]
                edited = [val for val in edited  if val in prior] + [val for val in prior if val not in edited]
                self.order[indep_key] = edited
        
        pass


    def shape(self):
        return [len(self.order[key]) for key in self.independent_keys] + [len(self.dependent_keys)]
    
    def get_val(self, *indices):
        dependent_key = self.dependent_keys[indices[-1]]
        independent_keys = {key:self.order[key][idx] for key, idx in zip(self.independent_keys, indices)}
        try:
            ret = self._f(independent_keys).get(dependent_key)
        except Exception:
            ret = None
        return ret
    
    def _replace(self, key):
        return str(self.label_dict.get(key, key))

    def get_label(self, axis, index=None):
        if axis == len(self.independent_keys):
            if index is None:
                return ""
            return self._replace(self.dependent_keys[index])
        else:
            if index is None:
                return self._replace(self.independent_keys[axis])
            return self._replace(self.order[self.independent_keys[axis]][index])

    
    
