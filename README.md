# PythonQOL

Written by Samuel Evans.  
*Questions, comments, or interested in contributing? Email sevans7@bu.edu .*  
**This repository is intended to provide quality-of-life improvements when coding in python.**  
See also the [PythonQOL Wiki](https://github.com/Sevans711/PythonQOL/wiki). (Click the link or the Wiki tab on github.)

## CONTENTS:  
- [QOL.plots: plotting (mainly matplotlib.pyplot)](https://github.com/Sevans711/PythonQOL/wiki/QOL.plots)  
- [QOL.files: storage  (mainly h5py)](https://github.com/Sevans711/PythonQOL/wiki/QOL.files)  
- [QOL.codes: other miscellaneous QOL code](https://github.com/Sevans711/PythonQOL/wiki/QOL.codes)  

## GETTING STARTED:  
### To copy this repository to your computer do (in terminal / command line):  
```
cd Dir   
git clone https://github.com/Sevans711/PythonQOL/
```
_(Replace `Dir` with the directory you want this folder to go in.  
This command will create a folder Dir/PythonQOL and then put the repository contents in that folder.  
One example is: `cd /Users/YourUserName/Desktop`)_  

### To "install" the files:
#### Choice 1 - "pip install" method. (Recommended)
```
cd PythonQOL
pip install .
```
This has the benefit of being relatively simple, and also if you ever want to uninstall you can just type `pip uninstall QOL`.

#### Choice 2 - "pythonpath" method.
Add the `PythonQOL` folder (if you're following these steps, it will be at the location `Dir/PythonQOL`) to your PYTHONPATH. _(The PATH tells python where it's allowed to search for File when you type "import File".)_  
As an example of how to do this: I use Spyder, so in the top left of my screen when spyder is open, I can click `python > PYTHONPATH manager`, then `+Add Path`. For other python consoles you may need to determine separately how to edit your PATH variable.  

#### _Troubleshooting: try relaunching your python shell/compiler after you have done the installation steps above._

### To start using the files:
Once you've completed choice 1 or choice 2 above, run this code to get started:
```python
import QOL.files as fqol
import QOL.plots as pqol
import QOL.codes as cqol
```

### To learn what's inside the files, here are some options:
- Check out the [PythonQOL Wiki](https://github.com/Sevans711/PythonQOL/wiki). (Click the link or the Wiki tab on github.)
- Use `help(obj)` to read the documentation inside the code. For example:
  - `help(pqol)` will print info about all the functions inside `PlotQOL` _(once you have done `import PlotQOL as pqol`)_.
  - `help(pqol.hline)` will info about the `hline()` function from `PlotQOL`.
  
## DEPENDENCIES:  
You may need to install `h5py` to run `FileQOL`.  
For that I direct you to http://docs.h5py.org/en/stable/build.html#  

_(TODO: add dependencies appropriately to setup.py so they are installed automagically)_

## DISTRIBUTION:  
Roughly, feel free to use/distribute/alter the code and any results from it as you see fit.  
Officially, see the [Liscense Page](https://github.com/Sevans711/PythonQOL/blob/master/LICENSE)


