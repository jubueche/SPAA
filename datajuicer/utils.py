import inspect
import copy
import concurrent.futures
import datajuicer.parser as parser
import random
import os.path
import datajuicer.in_out as in_out
import datajuicer.database as database
import time

#run_mode in normal, force, load, load_any

#cache_mode in normal, no_save
class NOMODEL:
    pass

def run(grid, func,n_threads=1, run_mode="normal", cache_mode="normal", cache_dir="Sessions/", store_key=None):
    def _runner(grid, *args, **kwargs):
        def _run(data, func, args, kwargs):
            if type(func) == str:
                func = get(data,func)
                assert callable(func)
            for i, arg in enumerate(args):
                if type(arg) is str:
                    args[i] = format_template(data, arg)
            for kw, arg in kwargs.items():
                if type(arg) is str:
                    kwargs[kw]= format_template(data, arg)
            if not hasattr(func, "table_name"):
                return func(*args,**kwargs)
            
            boundargs = inspect.signature(func).bind(*args,**kwargs)
            boundargs.apply_defaults()

            dependencies = {}
            for i, dep_name in enumerate(func.dependencies):
                assert type(dep_name) is str
                if ":" in dep_name:
                    arg, key = dep_name.split(":", 1)
                    if key.isnumeric():
                        key = int(key)
                    reserved_keys = [name.split(":",1) for name in func.dependencies]
                    reserved_keys.pop(i)
                    reserved_keys = [item for sublist in reserved_keys for item in sublist]
                    if not key in reserved_keys:
                        dep_name = key
                    dependencies[dep_name] = get(boundargs.arguments[arg], key)
                else:
                    dependencies[dep_name] = boundargs.arguments[dep_name]

            session_ids = database.select(db_file = os.path.join(cache_dir, "sessions.db"),column = "session_id", table = func.table_name, where= dependencies, order_by="start_time")
            sid = None
            for session_id in session_ids:
                checker = func.checker
                if checker is None:
                    checker = lambda *args: True
                if checker(session_id, func.table_name, cache_dir):
                    sid = session_id
                    break
            if sid is None or run_mode == "force":
                if run_mode == "load":
                    raise Exception("No Sessions Found")
                if run_mode == "load_any":
                    return NOMODEL
                random.seed()
                sid = random.randint(1000000000, 9999999999)
                data["session_id"] = sid
                row = copy.copy(dependencies)
                if cache_mode == "normal":
                    row["session_id"] = sid
                    row["start_time"] = int(time.time()*1000)
                    database.insert(db_file = os.path.join(cache_dir, "sessions.db"), table = func.table_name, row = row, primary_key = "session_id")
                ret = func(*args,**kwargs)
                if cache_mode != "no_save" and not func.saver is None:
                    func.saver(sid,func.table_name,cache_dir,ret)
            
            elif not func.loader is None:
                ret = func.loader(sid, func.table_name, cache_dir)
            
            if store_key is None:
                return ret
            elif store_key == "*":
                data.update(ret)
                return data
            else:
                data[store_key]=ret
                return data

        if type(grid) is dict:
            grid = [grid]
        
        if n_threads==1:
            ret = [_run(copy.copy(data),func,list(args),kwargs) for data in grid]
            
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                param_list = [(copy.copy(data),func,list(args),kwargs) for data in grid]
                futures = [executor.submit(_run, *param) for param in param_list]

            ret = [f.result() for f in futures] 
        return [r for r in ret if not r is NOMODEL]
    return lambda *args, **kwargs: _runner(grid, *args, **kwargs)
        

def cachable(dependencies=None, saver=in_out.easy_saver, loader=in_out.easy_loader, checker=in_out.easy_checker, table_name=None):
    def decorator(func, dependencies, saver, loader, checker, table_name):
        if table_name is None:
            table_name = func.__name__
        if dependencies is None:
            dependencies = list(inspect.signature(func).parameters)
        func.dependencies = dependencies
        func.saver = saver
        func.loader = loader
        func.checker = checker
        func.table_name = table_name
        return func
    return lambda func: decorator(func, dependencies, saver, loader, checker, table_name)

def format_template(data, template):
        if parser.in_brackets(template):
            return get(data, template[1:-1])
        else:
            return parser.replace(template, lambda k: str(get(data, k)))
class NODEFAULT:
    pass
def get(data, key, default=NODEFAULT):
    if key == "*":
        return data
    func_name, l_args = parser.get_arg_list(key)
    literal = False
    if func_name[0] == "!":
        literal = True
        func_name = func_name[1:]
    try:
        output = data[format_template(data, func_name)]
    except KeyError as e:
        if default is NODEFAULT:
            raise(e)
        else:
            output = default
    while type(output) is str and "{" in output:
        output = format_template(data, output)
    for args in l_args:
        for i,arg in enumerate(args):
            args[i] = format_template(data, arg)
        output = output(*args)
        if not literal and type(output) is str:
            output = format_template(data, output)
    return output

def split(grid, key, values, where={}):
    if type(grid) is dict:
        grid = [grid]
    
    output = []
    for data in grid:
        if all([get(data,kkey)==where[kkey] for kkey in where]):
            for val in values:
                copied = copy.copy(data)
                copied[key] = val
                output.append(copied)
        else:
            output.append(copy.copy(data))
    return output

def configure(grid, dictionary, where={}):
    if type(grid) is dict:
        grid = [grid]
    cg = [copy.copy(model) for model in grid]
    for model in cg:
        if all([get(model,key)==where[key] for key in where]):
            for key in dictionary:
                model[key] = dictionary[key]
    return cg


def query(grid, select, where={}, group_by=[], reduction=lambda x:x, return_func=False, flatten_reductions=False):
    if type(grid) is dict:
        grid = [grid]
    ret = []
    single_select = False
    if type(group_by) is str:
        group_by = [group_by]
    if type(select) is str:
        if select == "*":
            get_keys = lambda model: list(model.keys())
        else:
            select = [select]
            get_keys = lambda model: select
            if not flatten_reductions:
                single_select = True
    else:
        get_keys = lambda model: select
    
    if type(reduction) is dict:
        red_d = reduction
        reduction = lambda l: {key:red_d[key](l) for key in red_d}
    
    groups = []
    def get_index(model):
        for i, group in enumerate(groups):
            if all([get(model,k)==group[k] for k in group_by]):
                return i

    def prepare(model):
        model_group = {key:get(model,key) for key in group_by}
        i = get_index(model)
        if not i is None:
            return i
        
        groups.append(model_group)
        ret.append({key:[] for key in get_keys(model)})
        return len(groups) -1

    for model in grid:
        if all([get(model,key) == where[key] for key in where]):
            index = prepare(model)
            for key in get_keys(model):
                ret[index][key].append(get(model,key))
    
    ret = [{key: reduction(l) for key, l in group.items()} for group in ret]
    if flatten_reductions:
        for i,d in enumerate(ret):
            
            new_d = {}
            for key in d:
                if type(d[key]) is dict:
                    for k in d[key]:
                        new_d[key + "_" + k] = d[key][k]
                else:
                    new_d[key] = d[key]
            ret[i] = new_d
    
    if single_select:
        ret = [group[list(group.keys())[0]] for group in ret]
    if group_by == []:
        return ret[0]
    if return_func:
        return lambda group: ret[get_index(group)]
    return groups, ret

def reduce_keys(grid, keys, reduction, where=[], group_by=None):
    if group_by is None:
        if type(keys) is str:
            group_by = [k for k in grid[0].keys() if not k == keys]
        else:
            group_by = [k for k in grid[0].keys() if not k in keys]
        for data in grid:
            for i, key in enumerate(group_by):
                if not key in data:
                    del group_by[i]
    groups, q = query(grid, keys, where=where, group_by=group_by, reduction=reduction, return_func=False, flatten_reductions=True)
    return [{**group,**qe} for group, qe in zip(groups, q)]


def make_unique(grid):
    def try_to_serialize(value):
        try:
            return str(value)
        except:
            try:
                return json.dumps(value)
            except:
                try:
                    return pickle.dumps(value)
                except:
                    return ""
    sgrid = list(map(lambda model: "".join([try_to_serialize(pair) for pair in model.items()]), grid))
    ret = []
    for i in range(len(sgrid)):
        for j in range(i):
            if sgrid[j] == sgrid[i]:
                continue
        ret+=[copy.copy(grid[i])]
    return ret