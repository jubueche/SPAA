def replace(string, func, open_b='{', close_b='}'):
    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)
    if not open_b in string and not close_b in string:
        return string
    
    l = [0]
    depth = 0
    for i, char in enumerate(string):
        if char == open_b:
            depth+=1
            if depth==1:
                l.append(i)
                l.append(i+1)
        if char == close_b:
            depth-=1
            if depth ==0:
                l.append(i)
                l.append(i+1)
        if depth<0:
            raise Exception("Missing Opening Bracket")
    l.append(len(string))
    result = ""
    for i, (start, end) in enumerate(pairwise(l)):
        part = string[start: end]
        if i%2==0:
            result+=part
        else:
            result+=func(part)
    return result

def in_brackets(string,open_b="{", close_b="}"):
    if string[0] != open_b or string[-1]!= close_b:
        return False
    depth=0
    for char in string[1:-1]:
        if char == open_b:
            depth+=1
        if char == close_b:
            depth-=1
        if depth<0:
            return False
    return True

def get_arg_list(string):
    def p(string):
        end = None
        depth = 0
        if not string[0] =="(":
            raise Exception("Invalid Syntax")
        for i, char in enumerate(string):
            if char == '(':
                depth+=1
            if char == ')':
                depth-=1
                if depth ==0:
                    end=i
                    break
            if depth<0:
                raise Exception("Missing Opening Bracket")
        if end is None:
            raise Exception("Invalid Syntax")
        return string[1:end], string[end+1:]

    splitted = string.split("(",1)
    if len(splitted)==1:
        return string, []
    
    func_name, tail = splitted
    tail = "(" + tail
    l_args = []
    while tail!="":
        start, tail = p(tail)
        l_args.append(start.split(','))

    return func_name, l_args