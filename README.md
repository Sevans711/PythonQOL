# PythonQOL

Written by Samuel Evans.  
*Questions, comments, or interested in contributing? Email sevans7@bu.edu .*  
**This repository is intended to provide quality-of-life improvements when coding in python.**  
See also the [PythonQOL Wiki](https://github.com/Sevans711/PythonQOL/wiki). (Click the link or the Wiki tab on github.)

## DISTRIBUTION:  
Roughly, feel free to use/distribute/alter the code and any results from it as you see fit.  
Officially, see the [Liscense Page](https://github.com/Sevans711/PythonQOL/blob/master/LICENSE)

## CONTENTS:  
- [PlotQOL](https://github.com/Sevans711/PythonQOL/wiki/PlotQOL): plotting (mainly matplotlib.pyplot)  
- [FileQOL](https://github.com/Sevans711/PythonQOL/wiki/FileQOL): storage  (mainly h5py)  
- [CodeQOL](https://github.com/Sevans711/PythonQOL/wiki/CodeQOL): other miscellaneous QOL code  

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
#### Choice 1 - pip install .
After running the above lines, do:
```
cd PythonQOL
pip install .
```
This has the benefit of being relatively simple, and also if you ever want to uninstall you can just type `pip uninstall PythonQOL`.

#### Choice 2 - adding files to the pythonpath
Add the PythonQOL folder (which is currently at Dir/PythonQOL) to your PYTHONPATH. The PATH tells python where it's allowed to search for File when you type "import File".  
As an example of how to do this: I use Spyder, so in the top left of my screen when spyder is open, I can click `python > PYTHONPATH manager`, then `+Add Path`, then select the folder with this repository. For other python consoles you may need to determine separately how to edit your PATH variable. Make sure to relaunch spyder or your other compiler application

### To start using the files:
Once you've completed choice 1 or choice 2 above, run this code to get started:
```python
import FileQOL as fqol
import PlotQOL as pqol
import CodeQOL as cqol
```

### To learn what's inside the files, here are some options:
- Check out the [PythonQOL Wiki](https://github.com/Sevans711/PythonQOL/wiki). (Click the link or the Wiki tab on github.)
- Use `help(obj)` to read the documentation inside the code. For example:
  - `help(pqol)` will print info about all the functions inside `PlotQOL` _(once you have done `import PlotQOL as pqol`)_.
  - `help(pqol.hline)` will info about the `hline()` function from `PlotQOL`.
  
## DEPENDENCIES:  
You may need to install `h5py` to run `FileQOL`.  
For that I direct you to http://docs.h5py.org/en/stable/build.html#  

_(TODO: complete list of dependencies)_



