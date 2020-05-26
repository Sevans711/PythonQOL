#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:01:55 2020

@author: Sevans

Snippets of very useful/elegant code that can be non-trivial to write.
Check documentation of function for info on how to write the code yourself.
"""

#TODO: wrapper functions in general
#TODO: wrapper function for print input args & kwargs
#TODO: utilize wrapper/decorator to return local variables in case code crashes?

def flatten_once(l):
    """flattens list by one dimension. l must be a list of lists.
    Examples:
        [[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6].
        [['1','2','3']] -> ['1','2','3'].
        ['1',['2','3']] -> Error (because first element, '1', is not a list).
        [[1,2,3],[4,[5,6]],[ [[7],8],9 ]] -> [1,2,3,4,[5,6],[[7],8],9].
    Code:
        return [x for sublist in l for x in sublist]
    """
    return [x for sublist in l for x in sublist]
    
def attrs(Class):
    """returns all names of attributes of Class.
    Code:
        return Class.__dict__.keys()
    """
    return Class.__dict__.keys()

def pop_from_kwargs(popthis, kwargs, default=None):
    """pops popthis from kwargs, returning (popthis, remaining_kwargs)
    
    Removes key popthis from kwargs if possible.
    Returns (kwargs[popthis] or default if popthis not in kwargs, kwargs without popthis)
    
    Examples
    --------
    >>>def foo2(**kwargs):
    >>>    return cqol.pop_from_kwargs('x', kwargs, 7)
    >>>foo2(x=1, y=2, z=3)
    (1, {'y': 2, 'z': 3})
    >>>foo2(**foo2(x=1, y=2, z=3)[1])
    (7, {'y': 2, 'z': 3})
    
    Code:
    -----
        popped = kwargs.pop(popthis, default)
        return (popped, kwargs)
    """
    popped = kwargs.pop(popthis, default)
    return (popped, kwargs)

def strmatch(x, y):
    """returns whether x 'matches' y.
    
    A 'match' is equality, or leading/trailing *'s to stand in for anything.
    Cannot test for strings with non-special *'s in them.
    Only one of x & y is permitted to have '*'s in it. Else, returns -1.
    
    Examples
    --------------
    hello  matches   hello
     *lo   matches   hello
     he*   matches   hello
    *ll*   matches   hello
    *he*   matches   hello
    el* doesnt match hello
    *el doesnt match hello
    """
    if '*' in y: #then swap x & y.
        if '*' in x: return -1
        else: return strmatch(y, x)
    if x[0]=='*':
        if x[-1]=='*':
            return (x[1:-1] in y)
        else:
            return y.endswith(x[1:])
    elif x[-1]=='*':
        return y.startswith(x[:-1])
    else:
        return x==y

def strmatches(x, keep=None, discard=None):
    """returns which strings in x 'match' anything in keep, and nothing in discard.
    
    A 'match' is equality, or leading/trailing *'s to stand in for anything.
    Cannot test for strings with non-special *'s in them.
    
    For each string s in x, s will be in result iff
    s matches at least one string in keep AND matches no strings in discard.
    If  keep   is None, pretend s matches something in keep.
    If discard is None, pretend s matches nothing in discard.    
    
    Examples
    --------
    strmatches(['hi','hello','okay','ohhello'], keep=['hi'])
    >>> ['hi']
    strmatches(['hi','hello','okay','ohhello'], discard=['*hell*'])
    >>> ['hi', 'okay']
    strmatches(['hi','hello','okay','ohhello'], keep=['*o*'])
    >>> ['hello', 'okay', 'ohhello']
    strmatches(['hi','hello','okay','ohhello'], keep=['*o*'], discard="*hell*")
    >>> ['okay']
    """
    keep    =   [keep]  if  type(keep)  ==str else keep
    discard = [discard] if type(discard)==str else discard
    
    result = []
    ###Check matches with something in keep###
    if keep is None:
        result = list(x) #makes a copy of x
    else: #keep is not None
        for key in x:
            for s in keep:
                if strmatch(s, key):
                    result += [key]
                    break
    ###Check matches with nothing in discard###
    if discard is not None:
        keepers = []
        for key in result:
            keeping = True
            for s in discard:
                if strmatch(s, key):
                    keeping = False
                    break
            if keeping:
                keepers += [key]
        result = keepers
    ###return###
    return result
    
    
"""
#Simple example of using a wrapper function to do something if error occurs,
#   then propagate the error forward (without handling it).
#   Also properly maintains documentation of function via wraps.
#   To simplify example, can delete all lines labeled '#for documentation handling'.

from functools import wraps #for documentation handling
def sympathize(f):
    '''wraps f'''
    @wraps(f) #for documentation handling
    def foo(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print("an error... that's rough, friend.")
            raise
    return foo

@sympathize
def sadden(param1=1, param2=2):
    '''raises a sad error'''
    raise Exception("just a mean error messing up your code")

sadden()

>>> sadden()
an error... that's rough, friend.
Traceback (most recent call last):

  File "<ipython-input-236-666529ff83a2>", line 1, in <module>
    sadden()

  File "<ipython-input-235-e7737866ab43>", line 4, in foo
    return f(*args, **kwargs)

  File "<ipython-input-235-e7737866ab43>", line 12, in sadden
    raise Exception("just a mean error messing up your code")

Exception: just a mean error messing up your code
#"""

"""
#Simple example of using a wrapper function to reset an input list to
#   its initial state, if an error occurs.

def savemylist(f):
    def foo(input_list, *args, **kwargs):
        save = input_list.copy()
        try:
            return f(input_list, *args, **kwargs) #Note every function returns something.
                    #even without explicit 'return' statement, still returns None.
        except:
            print("an error! input was changed to:",input_list,
                  "before the error occured,\n but I am restoring its intial state.")
            for i in range(len(input_list)):
                input_list[i]=save[i]
            del save #explicitly free up this memory
    return foo

@savemylist
def messuplist(x, make_err=False):
    x[1]-=100
    if make_err:
        raise Exception("I made an error appear.")
    x[2]-=100

x=[0,1,2,3,4] #use np.array for python 3.2 or lower, if list.copy() not implemented.
    
>>> messuplist(x, make_err=False)
>>> print(x)
[0, -99, -98, 3, 4]
>>> messuplist(x, make_err=True)
an error! input was changed to: [0, -199, -98, 3, 4] before the error occured,
but I am restoring its intial state.
>>> x
[0, -99, -98, 3, 4]
#"""
