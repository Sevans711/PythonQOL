#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:16:36 2020

@author: Sevans
    
#Useful and user-friendly/convenient codes for file storage in h5py.
"""

#Investigate astropy tables
#Investigate pandas
#Investigate pickle for saving

#TODO: implement a function that tells you all available files in default h5dir.
#TODO: finish implementing fqol.update(file, x)
#      to add information to a file. x should be a dict().
#TODO: add multilayer-dict support to writedict() & update().


import h5py
import os

DEFAULT_H5DIR='/h5dir/'

### SET DEFAULT STORAGE LOCATION FOR h5 FILES
def set_h5dir(new_h5dir=None, DEFAULT=DEFAULT_H5DIR):
    """sets h5dir to the new_h5dir, or default if None is passed.
    
    h5dir is the default location for h5 file storage
    it defaults to os.getcwd()+'/h5dir/'.
    access via: import QOL.files as fqol; fqol.h5dir
    """
    global h5dir
    global h5dir_is_default
    if new_h5dir is None:         #set to default.
        h5dir = os.getcwd()+DEFAULT
        h5dir_is_default = True
    else:                           #set to inputted h5dir
        h5dir = new_h5dir
        h5dir_is_default = False
    return h5dir

#SET DEFAULT h5dir HERE#
h5dir_is_default = locals()['h5dir_is_default'] if 'h5dir_is_default' in locals().keys() \
                else True

if h5dir_is_default or 'h5dir' not in locals().keys():
    set_h5dir()
    
    
### BEGIN FUNCTION DEFINITIONS

def convert_to_filename(name, folder=h5dir):
    return name if name.startswith('/') else folder + name
    
def write(x, file, folder=h5dir, d="data", Verbose=True, overwrite=False):
    """Writes x to file.
    folder = defaults to h5dir; string ending with '/'.
    d = name of data in file; defaults to 'data'.
    overwrite = whether to overwrite f[d] if it already exists.
        (overwrite not implemented for writedict.)
    Pro tip: set d="data/subgroup" or other path, as desired.
    Automatically creates folder if folder does not exist but its enclosing folder does.
    """
    fname = convert_to_filename(file, folder)
    if Verbose: print("Writing",d,"to",fname)
    if not os.path.isdir(folder): os.mkdir(folder) #make h5dir if it doesn't exist already.
    
    f = h5py.File(fname,'a')
    if type(x)==dict:
        writedict(x, f, d=d, Verbose=Verbose, overwrite=overwrite)
    elif d in f:
        if overwrite:
            if Verbose: print("Overwriting",d)
            del f[d]
        else:
            if Verbose: print("failed to write to",d,"because it already exists")
            f.close()
            return
    else:
        f[d] = x
    f.close()
    
def writedict(x, f, d="data", Verbose=True, overwrite=False):
    """writes x to (the open h5py.File) f, saving under dataset d.
    x should be a dict of (possibly sub-dicts, eventually of) arrays.
    if x[key] is a dict, recurses, calling writedict again.
    helper function to 'write'.
    """
    for key in x.keys():
        dkey = d+"/"+str(key)
        if type(x[key])==dict:
            writedict(x[key], f, d=dkey)
        elif dkey in f:
            if overwrite:
                if Verbose>1: print("Overwriting",dkey)
                del f[dkey]
                f[dkey] = x[key]
            else: 
                if Verbose: print("failed to write to",dkey,"because it already exists")
        else:
            f[dkey] = x[key]
            
def update(f, x, d="data", folder=h5dir, Verbose=False):
    """updates data in f (str of h5file name) via dict.update().
    x should be a dict.
    only checks first layer of keys in f.
    (e.g. if f=dict(q=dict(a=3, b=5), then
    update(f, dict(q=dict(c=7))) would delete a=3 & b=5, in current implementation.
    """
    y = read(f, d=d, folder=folder, Verbose=(Verbose>1))
    y.update(x)
    write(y, f, d=d, folder=folder, Verbose=Verbose, overwrite=True)
    return y
    
def read(file, d="data", folder=h5dir, Verbose=False):
    """Reads data from file.
    folder = defaults to h5dir; string ending with '/'.
    d = name of data in file; defaults to 'data'.
    """
    fname = convert_to_filename(file, folder)
    if Verbose: print("Reading",d,"from",fname)
    
    if not isFile(fname):
        print: ("File",fname,"does not exist in readable form.")
        return None
    
    f = h5py.File(fname, 'r')
    if isinstance(f[d], h5py.Group):
        x = readdict(f)
    else:
        x = f[d][()] #[()] reads & saves info.
    f.close()
    return x

def readdict(f, d="data"):
    """Reads data from (the open h5py.File) f, returning result.
    f[d] should be a h5py.Group.
    helper function to 'read'."""
    x = dict()
    for key in f[d].keys(): 
        dkey = d+"/"+str(key)
        if isinstance(f[dkey], h5py.Group):
            readdict(f, dkey)
        else:
            x[key]=f[dkey][()] #[()] reads & saves info.
    if not "filename" in x.keys():
        x["filename"]=f.filename
    return x

def contents(file, folder=h5dir, d="data"):
    """returns list of all groups in file"""
    class store_during_run:
        def __init__(self):
            self.storage = []
        def store(self, x):
            self.storage += [x]
    s = store_during_run()
    f = h5py.File(convert_to_filename(file, folder), 'r')
    f.visit(s.store)
    f.close()
    return s.storage

def contains(file, x, folder=h5dir, d="data", Verbose=True):
    """returns whether file contains the group named d/x.
    e.g. if d=="data", x=="ux12", returns whether file contains data/ux12.
    if x starts with d+"/", assumes d is included already.
    e.g. if d=="d", x=="d/X/Y", checks for d/X/Y. (does not search for d/d/X/Y)
    """
    if not isFile(folder+file):
        if Verbose: print("Warning,",folder+file,"does not exist")
        return False
    def find_x(name):
        if x.startswith(d+"/"):
            if x ==name: return True
        elif d+"/"+x == name: return True
    f = h5py.File(folder+file, 'r')
    search = f.visit(find_x)
    f.close()
    return (search is not None)
    
def close_all():
    """Closes all hdf5 file objects in current namespace."""
    from gc import get_objects
    for obj in get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                s = str(obj)
                obj.close()
                if str(obj) != s:
                    print("Closed", s)
            except:
                pass
    print(" >> done")
    
def isFile(filename):
    return os.path.isfile(filename)

#FindFile:
"""
import glob
glob.glob('/Users/Sevans/Desktop/GapPrograms/Ringel_MapColorTheorem/**.py', recursive=True)
"""
