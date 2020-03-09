# PythonQOL
Written by Samuel Evans
Questions, comments, or interested in contributing? Email sevans7@bu.edu .
This repository is intended to provide quality-of-life improvements when coding in python.

DISTRIBUTION:
Feel free to use/distribute/alter the code and any results from it as you see fit,
as long as you do not claim explicity or implicitly that you are this code's creator.
(There's probably some sort of github liscense which says that,
but I am still new to github so please bear with me.
If you know about github liscenses and are willing to help, I am happy to accept advice!)

CONTENTS:
PlotQOL: plotting (mainly matplotlib.pyplot)
FileQOL: storage  (mainly h5py)
CodeQOL: other miscellaneous QOL code.

GETTING STARTED:
To copy this repository to your computer do (in terminal / command line) :
cd <Dir>      # where <Dir> = Directory you want this folder to go in, e.g. cd /Users/YourUserName/Desktop 
git clone https://github.com/Sevans711/PythonQOL/
And that's it!
To start using the files, I recommend adding the folder containing them to your python PATH.
For example, I use Spyder, so in the top left of my screen when spyder is open,
I can click python > PYTHONPATH manager, then +Add Path, select the folder with this repository.
For other python consoles you may need to determine separately how to edit your PATH variable.
The PATH tells python where to search for File when you type "import File".
Once you've added the folder with this repository to your PATH, you can do something like:
import FileQOL as fqol
import PlotQOL as pqol
import CodeQOL as cqol
#then to get started and check out some of the available functions without reading all the code, you could do
#help(obj) to read the documentation. e.g.:
help(pqol) #will print documentation about pqol, including what functions it contains and what they do.





