#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:01:55 2020

@author: Sevans

Snippets of very useful/elegant code that can be non-trivial to write.
Check documentation of function for info on how to write the code yourself.
"""

#TODO: func to pull, store, and remove any **kwargs from given_list.
#TODO: wrapper functions in general
#TODO: wrapper function for print input args & kwargs
#Maybe see https://amaral.northwestern.edu/blog/function-wrapper-and-python-decorator
#TODO: dealing with exceptions, & the raise statement
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

"""
#Simple example of using a wrapper function to do something if error occurs,
#   then propagate the error forward (without handling it).

def sympathize(f):
    def foo(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print("an error... that's rough, friend.")
            raise
    return foo

@sympathize
def sadden():
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
