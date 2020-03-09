#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:01:55 2020

@author: Sevans

Snippets of very useful/elegant code that can be non-trivial to write.
Check documentation of function for info on how to write the code yourself.
"""

#TODO: func to pull, store, and remove any **kwargs from given_list.

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