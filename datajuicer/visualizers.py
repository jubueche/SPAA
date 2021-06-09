from datajuicer.table import Table

def visualizer(dim=None):
    def _func(func, table_args, kwargs):
        table = Table(*table_args)
        return func(table, **kwargs)
    return lambda func: lambda grid, independent_keys, dependent_keys, label_dict={}, order=None, **kwargs: _func(func, [grid, independent_keys, dependent_keys, label_dict, dim, order], kwargs)

@visualizer(dim=4)
def latex(table, decimals=2, bold_order=None):
    shape = table.shape()
    

    def format_value(val):
        if type(val) is list:
            val = val[0]
        if type(val) is float:
            return f"%.{decimals}f" % (val)
        elif type(val) is str:
            return val.replace("_", r"\_")
        return str(val)

    relevant = [set([]) for _ in range(shape[0])]
    diff = [None] * shape[0]
    for i0 in range(shape[0]):
        for i1 in range(shape[1]):
            for i2 in range(shape[2]):
                for i3 in range(shape[3]):
                    if not format_value(table.get_val(i0, i1, i2, i3)) == "None":
                        relevant[i0].add(i1)
        relevant[i0] = sorted(list(relevant[i0]))
        diff[i0] = shape[1] - len(relevant[i0])
    
    mdiff = min(diff)

    cols = ("lc" + "c".join(["l"*shape[3]]*(shape[1]-mdiff)))
    string = r"\resizebox{\columnwidth}{!}{%" + "\n"
    string += r"\begin{tabular}{" + cols + "}\n"
    string += r"\toprule" + "\n"

    for i0 in range(shape[0]):
        struts = r""
        if i0 > 0:
            string += r"\midrule" + " \n"
        string += r"\multicolumn{" + str(len(cols)) + r"}{l}{\bfseries " \
            + format_value(table.get_label(axis=0, index=i0)) \
            + r"}" + struts + r"\\" + "\n"

        

        padding = " & " * ((diff[i0] - mdiff) * (shape[3] + 1))

        string += "".join([r"&& \multicolumn{" + str(shape[3]) +r"}{l}{" + format_value(table.get_label(axis=1, index=i)) + r"} " for i in relevant[i0]]) \
            + padding + r"\\" \
            + "".join([r"\cmidrule(r){" + f"{i*(shape[3]+1) +3}-{(i+1)*(shape[3]+1) + 1}" + "}" for i in range(len(relevant[i0]))]) \
            + "\n"
        
        if shape[3] > 1:
            string += format_value(table.get_label(axis=2)) \
                + (" && " + " & ".join([format_value(table.get_label(axis=3, index=i)) for i in range(shape[3])]) ) * len(relevant[i0]) \
                + padding + r" \\" + "\n"
        
        for i2 in range(shape[2]):
            vals = [[format_value(table.get_val(i0, i1, i2, i3)) for i3 in range(shape[3])] for i1 in relevant[i0]]
            vals = [[vv if not vv=='None' else '-1' for vv in v] for v in vals]
            if not all([all([val =='None' for val in v]) for v in vals]):
                if bold_order is None:
                    val_bold = [max([float(v[i]) for v in vals]) for i in range(len(vals[0]))]
                    bold_vals = [[("%.2f" % float(v)) for idx,v in enumerate(vv)] for vv in vals]
                else:
                    val_bold = [bold_order[i]([float(v[i]) for v in vals]) for i in range(len(vals[0]))]
                    bold_vals = [[f"{v}" if float(v) != val_bold[idx] else r"\bf{"+str(v)+r"}" for idx,v in enumerate(vv)] for vv in vals]
                string += format_value(table.get_label(axis=2, index=i2)) \
                    + "".join([" && "  + " & ".join(v) for v in bold_vals]) \
                    + padding + r"\\" + "\n"

    string += r"\bottomrule" + "\n"
    string += r"\end{tabular}%" + "\n"
    string += "}"
    return string