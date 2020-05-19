# PythonQOL

Written by Samuel Evans.  
*Questions, comments, or interested in contributing? Email sevans7@bu.edu .*  
**This repository is intended to provide quality-of-life improvements when coding in python.**  
See also the [PythonQOL Wiki](https://github.com/Sevans711/PythonQOL/wiki). (Click the link or the Wiki tab on github.)

## DISTRIBUTION:  
Feel free to use/distribute/alter the code and any results from it as you see fit,
as long as you do not claim explicity or implicitly that you are this code's creator.
(There's probably some sort of github liscense which says that,
but I am still new to github so please bear with me.
If you know about github liscenses and are willing to help, I am happy to accept advice!)

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
One example is: `cd /Users/YourUserName/Desktop`)_  
And that's it!  
### To start using the files:
1. I recommend adding the folder containing them to your python PATH. The PATH tells python where it's allowed to search for File when you type "import File".  
As an example of how to do this: I use Spyder, so in the top left of my screen when spyder is open, I can click `python > PYTHONPATH manager`, then `+Add Path`, then select the folder with this repository. For other python consoles you may need to determine separately how to edit your PATH variable.  _(TODO: provide more detailed support for this step)_

2. Once you've added the folder with this repository to your PATH, you can do something like:  
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



