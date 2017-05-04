"""6.009 Lab 8A: carlae Interpreter"""

import sys


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


#####################################################################################################
#####################################################################################################
#####################################################################################################


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    return_list = []
    # keep track of whether a current line being read is a comment
    is_comment = False
    current_word = ""
    
    for char in source:
        if is_comment:
            # if comment is being read, pass until newline char is read,
            #in which case comment has ended
            if char == '\n':
                is_comment = False
            pass
        else:
            if char == '(':
                return_list.append(char)
            elif char == ')':
                if current_word != "":
                    #if char is ) and there is a word being read, then
                    #this is the end of tha word
                    return_list.append(current_word)
                    current_word = ""
                return_list.append(char)
            elif char == ";":
                #next chars in this line will be comments
                is_comment = True
            elif char.isspace():
                if current_word != "":
                    #if char is white space and there is a word being read, then
                    #this is the end of that word
                    return_list.append(current_word)
                    current_word = ""
            else:
                #add char to word being read
                current_word = current_word + char

    #if there is still a word read, add it to the list of tokens 
    if current_word != "":
        return_list.append(current_word)


    return return_list


#####################################################################################################
#####################################################################################################
#####################################################################################################


#helper function
def get_inner_expr(tokens):
    """
    find inner expressions within an expression given the tokens for said expression

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    #keeps count of open parens
    return_list = []
    paren_count = 0
    start_index = 0
    end_index = 0
    for i in range(len(tokens)):
        t = tokens[i]
        if t == "(":
            #paren has been opened
            paren_count += 1
            if paren_count == 2:
                #if two open parens, then this is the start of an inner expression
                start_index = i
        if t == ")":
            #paren has been closed
            paren_count += -1
            if paren_count == 1:
                #if only one paren, this is the end of an inner expression
                #add it to return list
                end_index = i
                return_list.append(tokens[start_index:end_index+1])

    #return inner expressions
    return return_list

#Helper function
def is_valid_entry(tokens):
    """
    Check to see if a list of tokens contains any syntax errors

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    #keep stack of markers of opened parens. When a paren is closed, remove top marker
    #if you cannot remove a marker, there is a syntax error
    paren_stack = []
    for t in tokens:
        if t == "(":
            paren_stack.append(1)
        if t == ")":
            if len(paren_stack) == 0:
                return False
            else:
                paren_stack.pop()

    #if there are any open parens that were not closed, there is a syntax error
    if len(paren_stack) != 0:
        return False
    return True

#helper function
def parse_token(token):
    """
    parse a token that is not a parenthesis
    ie if a token is a number, return its number value, else return the token

    Arguments:
        token (string): a string representing a token
    """
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    #raise SyntaxError if not a correct token list
    if not is_valid_entry(tokens):
        raise SyntaxError

    return_list = []
    started = False
    i = 0
    inner_expr_counter = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "(":
            if not started:
                #if this is the first (, then new first grouping has started
                started = True
                i += 1
            else:
                #this is the start of an inner expression
                #get next inner expression and add to return list
                inner_expr = get_inner_expr(tokens)[inner_expr_counter]
                return_list.append(parse(inner_expr))
                #increment counter by len of inner_expr to skip over it
                i += len(inner_expr)
                inner_expr_counter += 1
        elif token == ")":
            #if ) is found, this is the end of a grouping
            break
        else:
            #else regular token, add it to return list
            return_list.append(parse_token(token)) 
            i += 1
    
    if len(return_list) < 2 and not started:
        #if only one ele and no groupings have started, then this is a stand alone token
        return return_list[0]
    return return_list


#####################################################################################################
#####################################################################################################
#####################################################################################################

#Lab 8B 


class LinkedList:
    def __init__(self, elt=None, next_elt=None):
        self.elt = elt
        self.next = next_elt


def list_func(args):
    if len(args)==0:
        return LinkedList()

    prev_list_node = None
    for arg in reversed(args):
        prev_list_node = LinkedList(arg, prev_list_node)
    return prev_list_node

def get_list_length(arg):
    if arg.elt == None:
        return 0
    counter = 0
    next_elt = arg
    while next_elt != None:
        counter += 1
        next_elt = next_elt.next
    return counter

def elt_at_index(linkedlist, index):
    if get_list_length(linkedlist) <= index:
        raise EvaluationError
    counter = index
    current_elt = linkedlist
    while counter != 0:
        current_elt = current_elt.next
        counter += -1
    return current_elt.elt

def car(linkedlist):
    if linkedlist.elt == None:
        raise EvaluationError
    return linkedlist.elt

def cdr(linkedlist):
    if linkedlist.elt == None:
        raise EvaluationError
    return linkedlist.next

def concat(lists):
    if len(lists)==0:
        return LinkedList()

    prev_list_node = None
    for arg in reversed(lists):
        for i in reversed(range(get_list_length(arg))):
            elt = elt_at_index(arg, i)
            prev_list_node = LinkedList(elt, prev_list_node)

    return prev_list_node

def map_func(func, linkedlist):
    list_elts = []
    if type(func) == LambdaObject:
        for i in range(get_list_length(linkedlist)):
            list_elts.append(evaluate([func , elt_at_index(linkedlist, i)],func.environment))
    else:
        for i in range(get_list_length(linkedlist)):
            list_elts.append(func([elt_at_index(linkedlist, i)]))

    return list_func(list_elts)

def filter_func(func, linkedlist):
    list_elts = []
    if type(func) == LambdaObject:
        for i in range(get_list_length(linkedlist)):
            filtered_elt = evaluate([func, elt_at_index(linkedlist, i)],func.environment)
            if filtered_elt == '#t':
                list_elts.append(elt_at_index(linkedlist, i))
    else:
        for i in range(get_list_length(linkedlist)):
            filtered_elt = func([elt_at_index(linkedlist, i)])
            if filtered_elt == "#t":
                list_elts.append(elt_at_index(linkedlist, i))
    
    return list_func(list_elts)

def reduce_func(func, linkedlist, initial_val):
    prev_val = initial_val
    if type(func) == LambdaObject:
        for i in range(get_list_length(linkedlist)):
            prev_val = evaluate([func, prev_val, elt_at_index(linkedlist, i)],func.environment)
    else:
        for i in range(get_list_length(linkedlist)):
            prev_val = func([prev_val, elt_at_index(linkedlist, i)])
    
    return prev_val


#####################################################################################################

def evaluate_file(file_name, environment=None):
    if environment == None:
        environment = Environment({}, built_in_envi)

    file_opener = open(file_name)

    read_file = file_opener.read()
    return evaluate(parse(tokenize(read_file)), environment)


#####################################################################################################
#####################################################################################################
#####################################################################################################


# Helper functions for built-in symbols
def mult(args):
    total = 0
    for i in args:
        if total == 0:
            total = i
        else:
            total = total*i
    return total

def div(args):
    total = 0
    for i in args:
        if total == 0:
            total = i
        else:
            total = total/i
    return total

def equals(args):
    prev_arg = None
    for arg in args:
        if prev_arg == None:
            prev_arg = arg
        else:
            if arg != prev_arg:
                return '#f'
            prev_arg = arg
    return '#t'

def decreasing(args):
    prev_arg = None
    for arg in args:
        if prev_arg == None:
            prev_arg = arg
        else:
            if not (prev_arg > arg):
                return '#f'
            prev_arg = arg
    return '#t'

def nonincreasing(args):
    prev_arg = None
    for arg in args:
        if prev_arg == None:
            prev_arg = arg
        else:
            if not (prev_arg >= arg):
                return '#f'
            prev_arg = arg
    return '#t'

def increasing(args):
    prev_arg = None
    for arg in args:
        if prev_arg == None:
            prev_arg = arg
        else:
            if not (prev_arg < arg):
                return '#f'
            prev_arg = arg
    return '#t'

def nondecreasing(args):
    prev_arg = None
    for arg in args:
        if prev_arg == None:
            prev_arg = arg
        else:
            if not (prev_arg <= arg):
                return '#f'
            prev_arg = arg
    return '#t'

#built in functions
carlae_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),

    '*': mult,
    '/': div,
    '=?': equals,
    '>': decreasing,
    '>=': nonincreasing,
    '<': increasing,
    '<=': nondecreasing,
    'not': lambda args: '#f' if args[0]=='#t' else '#t',
    '#t': '#t',
    '#f': '#f',
    'list': list_func,
    'car': lambda arg: car(arg[0]),
    'cdr': lambda arg: cdr(arg[0]),
    'length': lambda arg: get_list_length(arg[0]),
    'elt-at-index': lambda args: elt_at_index(args[0], args[1]),
    'concat': concat,
    'map': lambda args: map_func(args[0], args[1]),
    'filter': lambda args: filter_func(args[0], args[1]),
    'reduce': lambda args: reduce_func(args[0], args[1], args[2]),
    'begin': lambda args: args[-1]
}

class Environment:
    """
    Environment structure

    Attributes:
        bindings (dict): dictionary with variable as key and its corresponding binding as value
        parent (Environment): parent of this Environment

    Methods:
        get_var:
            Arguments:
                var (string): variable to get value of
            Returns:
                value of var, if it exists, else raises EvaluationError
        get_var_environment:
            Arguments:
                var (string): variable to get binding environment of
            Returns:
                environment in which var is binded, else raises EvaluationError
        is_var_in_environment:
            Arguments:
                var (string): variable to see if it is in Environment or ancestor Environment
            Returns:
                True if in Environment or ancestor Environment, otherwise False
        set_binding:
            Arguments:
                var (string): variable to set binding
                val : value to set variable to
            Returns:
                the value set to the variable
    """
    def __init__(self, bindings = {}, parent = None):
        self.bindings = bindings
        self.parent = parent

    def get_var(self, var):
        if var in self.bindings.keys():
            return self.bindings[var]
        if self.parent != None:
            return self.parent.get_var(var)
        else:
            raise EvaluationError

    def get_var_environment(self, var):
        if var in self.bindings.keys():
            return self
        if self.parent != None:
            return self.parent.get_var_environment(var)
        else:
            raise EvaluationError

    def is_var_in_environment(self, var):
        if var in self.bindings.keys():
            return True
        if self.parent != None:
            return self.parent.is_var_in_environment(var)
        else:
            return False

    def set_binding(self, var, val):
        self.bindings[var] = val
        return val

class LambdaObject:
    """
    Lambda object

    Attributes:
        params (list): list of parameters of lambda function
        expr (list): expression of lambda function
        environment (Environment): Environment in which lambda function was created

    """
    def __init__(self, params, expr, environment):
        self.params = params
        self.expr = expr
        self.environment = environment

#Initiate built in envoronment, and default environment
built_in_envi = Environment(carlae_builtins)

def evaluate(tree, environment = None):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    #if no environment passed in, create a new environment
    if environment == None:
        environment = Environment({}, built_in_envi)

    #if number, return number
    if type(tree) == int or type(tree) == float:
        return tree
        #if symbol in envi, return symbol val

    elif type(tree) == list:
        if len(tree) == 0:
            raise EvaluationError
        #else, is list
        #first token is operation
        operation = tree[0]
        operation = evaluate(tree[0], environment)
        
        #eval as special form
        #first three ifs are pretty simply boolean stuff
        if operation == "if":
            cond = evaluate(tree[1], environment)
            if cond == "#t":
                return evaluate(tree[2], environment)
            else:
                return evaluate(tree[3], environment)
        if operation == "and":
            for arg in tree[1:]:
                what_is_arg = evaluate(arg, environment)
                if what_is_arg == "#f":
                    return "#f"
            return "#t"
        if operation == "or":
            for arg in tree[1:]:
                what_is_arg = evaluate(arg, environment)
                if what_is_arg == "#t":
                    return "#t"
            return "#f"

        #eval as define
        if operation == "define":
            #if first arg is string, is regular definition
            if type(tree[1]) == str:
                #evaluate var val and the do define_func with it
                val = evaluate(tree[2], environment)
                environment.set_binding(tree[1], val)
                return val
            #else definition is "shortcut" version
            else:
                #convert "shortcut" to long form and evaluate that
                lambda_equivalent = ["define", tree[1][0], ["lambda", tree[1][1:] , tree[2]]]
                return evaluate(lambda_equivalent, environment)

        #create and return new LambdaObject
        #tree should be of form [lambda, [params], [expr]]
        if operation == "lambda":
            params = tree[1]
            expr = tree[2]
            return LambdaObject(params, expr, environment)

        #eval as let
        if operation == "let":
            vars_and_vals = tree[1]
            body = tree[2]
            temp_bindings = {}
            for pair in vars_and_vals:
                var = pair[0]
                val = evaluate(pair[1],environment)
                temp_bindings[var] = val
            temp_envi = Environment(temp_bindings, environment)
            return evaluate(body, temp_envi)

        #eval as set!
        if operation == "set!":
            var = tree[1]
            expr = evaluate(tree[2], environment)
            var_envi = environment.get_var_environment(var)
            return_val = var_envi.set_binding(var, expr)
            return return_val

        #eval as Lambda Object
        if type(operation) == LambdaObject:
            param_vals = tree[1:]
            param_stack = [p for p in param_vals]
            temp_bindings = {}
            lambda_envi = operation.environment
            for i in operation.params:
                temp_val = param_stack.pop(0)
                temp_val = evaluate(temp_val, environment)
                temp_bindings[i] = temp_val
            temp_envi = Environment(temp_bindings, lambda_envi)
            return evaluate(operation.expr, temp_envi)

        #else eval as a regular function
        else:
            func = operation
            if type(func) == LambdaObject:
                #if func is lambda func, treat it separately as follows
                return evaluate([func, tree[1:]], environment)

            if not callable(func):
                #if not func, incorrect input
                raise EvaluationError

            args_list = []
            for arg in tree[1:]:
                #eval args
                evalled_arg = evaluate(arg, environment)
                args_list.append(evalled_arg)

            #apply args to func
            val = func(args_list)
            return val

    #if tree is string, it is either special form or variable
    elif type(tree) == str:
        if tree=="if" or tree=="and" or tree=="or" or tree=="define" or tree=="lambda" or tree=="set!" or tree=="let":
            # these are special forms
            return tree
        if environment.is_var_in_environment(tree):
            #if variable exists, return its val
            var_val = environment.get_var(tree)
            return var_val
        else:
            #variable not found
            raise EvaluationError
    elif type(tree) == LambdaObject:
        return tree
    elif type(tree) == LinkedList:
        return tree
    else:
        #incorrect input type
        raise EvaluationError

def result_and_env(tree, environment=None):
    if environment == None:
        environment = Environment({}, built_in_envi)
    evalled = evaluate(tree, environment)
    return (evalled, environment)


#####################################################################################################
#####################################################################################################
#####################################################################################################


if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    envi = Environment({}, built_in_envi)

    files_to_eval = sys.argv[1:]
    if len(files_to_eval) > 0:
        for file_name in files_to_eval:
            evaluate_file(file_name, envi)
    
    expr = input("in> ")

    while expr != "QUIT":
        try:
            output = evaluate(parse(tokenize(expr)), envi)
            """
            new_output = []
            if type(output) == LinkedList:
                for i in range(get_list_length(output)):
                    elt = elt_at_index(output, i)

            """
            print("  out>", output)
        except SyntaxError:
            print("SyntaxError")
        except EvaluationError:
            print("EvaluationError")
        except Exception as err:
            print("UnexpectedError \n", err)

        expr = input("in> ")


