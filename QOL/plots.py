#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:52:21 2020

@author: Sevans

Useful and user-friendly/convenient codes for making matplotlib plots.
"""

#KNOWN ISSUES:
'''
KNOWN ISSUES:
#bbox_corners for text is not actually correct (but it is close to correct).
#for example, try the following with in-line plotting:
plt.figure()
plt.text(0.2, 0.6, 'hello there! how is it going?', fontsize=18)
to = pqol.texts_overlap(gridsize=(10,10))
pqol.imshow_overplot(to)
bbc = pqol.bbox_corners(pqol.get_texts()[0])
pqol.vline(bbc[0][0]) #draw vline at text bbox's left coord
pqol.vline(bbc[1][0]) #draw vline at text bbox's right coord
pqol.hline(bbc[0][1]) #draw hline at text bbox's bottom coord
pqol.hline(bbc[1][1]) #draw hline at text bbox's top coord
pqol.colorbar() #also try to remove this line and see how it changes
'''

#TODO: Config file for defaults...
#TODO: read default values in functions instead of at function definition.
    #this allows changing default values to impact function behavior
    #without requiring the module to be reloaded.
    #(also if function code wasn't changed then import QOL.plots will not be
    # sufficient to update default; requires restarting python session.)
#TODO: implement title for colorbar()
#TODO: implement scatter plot marker cycle. use cycler to do cycles?
#TODO: implement different left/right yscales to put two plots on same grid.
    #Can be accomplished by just doing:
    #plt.plot(<firstplot>); plt.twinx(); plt.plot(<secondplot>)
    #However there are QOL issues with labeling, colors, etc, that could be fixed.
#check out illustrator
#TODO: determine size of a textbox or legend before it is plotted - use that to inform how you plot.
#   ^^^^^ this one is highest priority ^^^^^
#TODO: option for text size as percentage of figure size.
#TODO: properly implement do_ylim for log-scale plots.
#TODO: increase dpi and decrease figure size? investigate...
#   see e.g.: https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
#TODO: add "spot" parameter to legend() & text(), & show in locs_visual().
#   works like badness parameter (&overwrites), except locations are fixed.
#TODO: implement max number of ticks for non-discrete colorbars.
#TODO: dynamically guess best step size for discrete imshow
#TODO: implement x label text overlap checker.
#   (prevent xticks from overlapping, e.g. when font is large.)
#TODO: std on vars in dictplot.
#TODO: improve efficiency of test_overlap and text_overlap.
#   instead of looping through the entire grid, only loop through those boxes
#   which may be close to the textbox / legend / box in question.
#TODO: improve pqol.text and pqol.legend documentation to show
#   the more sophisticated overlap-checking that is now implemented.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.stats import linregress
import os #only used for saving figures.
from matplotlib.colors import LinearSegmentedColormap #only use for discrete_cmap

from QOL.codes import strmatch #only used in pqol.dictplot
from QOL.codes import strmatches #only used in pqol.dictplot

DEFAULT_FIGSIZE=(6,6)       #for fixfigsize
DEFAULT_DPI=100             #for fixdpi
XYLIM_MARGIN=0.05           #for do_xlim, do_ylim
TEXTBOX_MARGIN=0.002        #for hline, vline
DEFAULT_SAVE_STR="Untitled" #for savefig
DEFAULT_GRIDSIZE=(16,16)    #(Nrows (y), Ncols (x)). for data_overlap
DEFAULT_SAVEDIR='/saved_plots/' #savefig saves to: os.getcwd()+DEFAULT_SAVEDIR
    #full directory name stored inpqol.savedir. Edit via pqol.set_savedir()


#### set better defaults ####

def fixfonts(s=12, m=15, l=18):
    """sets better default font sizes for plots"""
    plt.rc('axes', titlesize=l)    # fontsize of the axes title
    plt.rc('figure', titlesize=l)  # fontsize of the figure title
    plt.rc('axes', labelsize=l)    # fontsize of the x and y labels
    
    plt.rc('font', size=m)         # controls default text sizes
    plt.rc('legend', fontsize=m)   # legend fontsize
    
    plt.rc('xtick', labelsize=s)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)   # fontsize of the tick labels
    

def fixfigsize(size=DEFAULT_FIGSIZE):
    """sets better default figure size for plots"""
    plt.rcParams['figure.figsize'] = size
    
def fixdpi(dpi=DEFAULT_DPI):
    """sets better default figure dpi for plots"""
    plt.rcParams['figure.dpi'] = dpi
    
def set_plot_defaults():
    """sets better defauls (fonts & figsize) for plots"""
    fixfigsize()
    fixfonts()
    fixdpi()
    
set_plot_defaults() #actually sets the defaults upon loading/importing QOL/plots.py


#### slice string interpreter ####
    #Not intrinsically related to "plots" though. belongs in different QOL file?
    #Used mainly for iplot method, below.
def str2idx(string):
    """converts s into the indexing tuple it represents.
    
    If string includes brackets [], they will be ignored.
    Ellipses (...) are not currently supported.
    
    Examples
    --------
    >>> pqol.str2idx(":9")
    (slice(None, 9, None),)    
    >>> np.arange(20)[pqol.str2idx("3:9")]
    array([3, 4, 5, 6, 7, 8])         
    >>> np.arange(20).reshape(4,5)[pqol.str2idx(":,1")]
    array([ 1,  6, 11, 16])     
    >>> np.arange(20).reshape(4,5)[pqol.str2idx("::2,1")]
    array([ 1, 11]) 
    """
    ans = []
    dims = (string.replace('[','').replace(']','')).split(',')
    for slicestring in dims:
        s = slicestring.split(":")
        if len(s)==1:
            ans += [int(s[0])]
        else:
            ans += [slice( *[(int(x) if x!='' else None) for x in s] )]
    return tuple(ans)



#### plt.plot functionality ####

def iplot(x, y=None, ss=None, i=None, xfunc=lambda x: x, yfunc=lambda y: y,
          plotter=plt.plot, iplotter=None, **kwargs):
    """plots y vs x, both indexed by index array i or slice_string ss.
    
    (Returns the output of the plotting function.)
    
    Parameters
    ----------
    x, y : arrays with the data to be plotted.
        If y is None, plots x vs np.arange(len(x)).
    ss : None, or slice_string ss. e.g. ":9", ":,8,:", or "::3".
        Indexes both x and y. Plot will be of y[pqol.str2idx(ss)] vs x[pqol.str2idx(ss)].
        Check help(pqol.str2idx) for further documentation about ss formatting.
        If ss is None, try to index by 'i' instead.
    i : None, or list of integers or booleans, of length == len(x)
        Indexes both x and y. Plot will be of y[i] vs x[i].
        If i  is None (and ss is also None), plots y vs x.
    xfunc : function. Default: x->x
        applies xfunc to all xdata before plotting.
        xfunc must accept x (array-like object) as input.
    yfunc : function. Default: y->y
        applies yfunc to all ydata before plotting.
        yfunc must accept y (array-like object) as input.
    plotter : function
        Must accept x and y as args; label and **kwargs as kwargs.
    iplotter : None, or function
        Use iplotter instead of plotter if iplotter is passed.
        Useful mainly for using non-default plotter if iplot is passed as input
        to a method which already has 'plotter' as kwarg, such as pqol.dictplot.
        (e.g.: pqol.dictplot(d, plotter=pqol.iplot, iplotter=plt.scatter))
    remaining **kwargs go to plotter or iplotter if not None.
    
    Examples
    --------
    x = np.array([ 2, 4, 6, 8,10,12,14,16, 18])
    y = np.array([-7,-3,-1, 0, 0,-1,-3,-7,-16])
    #The following three lines are equivalent:
    pqol.iplot(x, y, ss="0::2")
    pqol.iplot(x, y, i =[0,2,4,6,8])
    pqol.iplot(x, y, i =[True,False,True,False,True,False,True,False,True])
    #Also try out the following:
    pqol.iplot(x, y, ss="3:8", plotter=plt.scatter)
    pqol.iplot(x, y, i= ((x >=2 ) & (x <=7)), plotter=plt.scatter)
    pqol.iplot(x, y, i= (y < -4), plotter=plt.scatter)
    """
    
    if y is None:
        y=x
        x=np.arange(len(x))
    plotter = plotter if iplotter is None else iplotter
    if ss is not None:
        s=str2idx(ss)
        return plotter(xfunc(x[s]), yfunc(y[s]), **kwargs)
    elif i is not None:
        return plotter(xfunc(x[i]), yfunc(y[i]), **kwargs)
    else:
        return plotter(xfunc(x), yfunc(y), **kwargs)
    
def dictplot(x, y=None, yfunc=lambda y:y, xfunc=lambda x:x,
             keys=None, hide_keys=None, labels='^', prefix='', suffix='', 
             plotter=plt.plot, legend_badness=0, stylize_keys=None, **kwargs):
    """plots all data from dict on one plot, using keys as labels.
    
    Parameters
    ----------        
    x : dict, list, or string which is a key of y
    y : None, or dict
        Here are the (only) valid (x,y) input combinations and what they do:
        (dict, None) -> plot each x[key] against np.arange(len(x[key])).
        (dict, dict) -> for each shared key in x & y, plot y[key] vs x[key].
        (list, dict) -> plot each y[key] against x.
        (str , dict) -> plot each y[key] against y[x] (except do not plot y[x]).  
    yfunc : function
        runs on all y-axis data before plotting. Default is y -> y
    xfunc : function
        runs on all x-axis data before plotting. Default is x -> x                             
    keys : None, or list of strings.
        Will only plot for key in keys and in dict.
        keys can use leading and/or trailing '*' as wildcard.
        e.g. keys=['*12', 'ux*'] includes all keys ending in '12' or starting with 'ux'.
    hide_keys : [], or list of strings.
        Will not show any key in hide_keys.
        keys can use leading and/or trailing '*' as wildcard.
        e.g. hide_keys=['*_1*', 'bz'] hides all keys containing '_1' or equal to 'bz'.
    labels : string. Default: '^'
        how to label keys. any '^' will be replaced with key name from dict.
    prefix : string
        prefix to all labels. (useful if plotting multiple dicts with same keys.)
    suffix : string
        suffix to all labels. (useful if plotting multiple dicts with same keys.)
    plotter : function
        Must accept x and y as args; label and **kwargs as kwargs.
    legend_badness : integer >= 0. Default 0.
        Badness in legend placement based on pqol.legend()
    stylize_keys : None or [str, style_dict] or [s1, sd1, ..., sN, sdN]. Default None.
        Stylize keys that match str, using style_dict.
        e.g.:
        stylize_keys=['ux*', dict(lw=1)]
            will change lw to 1 for keys starting with 'ux', only.
        stylize_keys=['ux*', dict(lw=1), '*b*', dict(lw=7)]
            lw=1 for keys starting with 'ux', and lw=7 for keys containing 'b'.
        stylize_keys=[['ux*','*12'], dict(ls='--'), 'ez', dict(color='blue')]
            lw=1 for keys starting with 'ux' or ending in '12', and
            color=blue for key equal to 'ez'.
    **kwargs are passed to plotter.
    
    Examples
    --------
    #import QOL.plots as pqol; then try the following:
    x  = np.array([ 2, 4, 6, 8,10,12,14,16, 18])
    y1 = np.array([-7,-3,-1, 0, 0,-1,-3,-7,-16])
    y2 = np.array([ 7, 3, 1, 0, 0, 1, 3, 7, 16])
    y3 = np.array([ 5, 5, 5, 5, 5, 5, 5, 5, 5 ])
    d  = dict(xData=x, y1=y1, ySecond=y2, y3rd=y3)
    y_all = np.concatenate([y1,y2,y3])
    xlims = [x.min()     -1, x.max()     +1]
    ylims = [y_all.min() -1, y_all.max() +1]
    pqol.dictplot(d)
    plt.title("plot A"); plt.show()
    pqol.dictplot("xData", d)
    plt.title("plot B"); plt.xlim(xlims); plt.ylim(ylims); plt.show()
    pqol.dictplot("xData", d, yfunc=lambda y: y/2)
    plt.title("plot C"); plt.xlim(xlims); plt.ylim(ylims); plt.show()
    pqol.dictplot("xData", d, hide_keys=["y2"], plotter=plt.scatter)
    plt.title("plot D"); plt.xlim(xlims); plt.ylim(ylims); plt.show()
    pqol.dictplot("xData", d, plotter=pqol.iplot, ss="3:8")
    plt.title("plot E"); plt.xlim(xlims); plt.ylim(ylims); plt.show()
    pqol.dictplot("xData", d, plotter=pqol.iplot, ss="3:8", iplotter=plt.scatter)
    plt.title("plot F"); plt.xlim(xlims); plt.ylim(ylims); plt.show()
    """
    ## SET UP KEYS TO PLOT ##
    if y is None:
        d = x
        keys = strmatches(d.keys(), keys, hide_keys)
        xvals = np.arange(len(yfunc(x[keys[0]]))) #may be inefficient for expensive yfunc.
        xvals = {key: (xvals) for key in keys}
    else:
        d = y
        keys = strmatches(d.keys(), keys, hide_keys)
        if type(x)==str:
            if x not in d.keys():
                print("Error, x (str) must be a key of y (dict).")
                return
            else:
                keys = [key for key in keys if ( key != x )]
                xvals = {key: (y[x]) for key in keys}
                #note: memory is fine; id(xvals[key_i])==id(xvals[key_j]).
        elif type(x)==dict:
            keys = [key for key in keys if ( key in d.keys() and key in x.keys() )]
            xvals = x
        else:
            xvals = {key: (x) for key in keys}
    
    ## PLOT AND STYLIZE KEYS ##       
    failed_to_plot_keys = []
    for key in keys:  
        
        if stylize_keys is None:
            kwargcopy = kwargs
        else:
            kwargcopy = {k:v for k,v in kwargs.items()}
            for i in range(len(stylize_keys)//2):
                (s_i, style_dict_i) = (stylize_keys[2*i], stylize_keys[2*i+1])
                s_i = [s_i] if type(s_i)==str else s_i
                for s in s_i:
                    if strmatch(s, key):
                        kwargcopy.update(style_dict_i)      
        try:
            plotter(xfunc(xvals[key]), yfunc(d[key]),
                    label=(prefix+labels+suffix).replace('^',key),
                    **kwargcopy)
        except:
            failed_to_plot_keys += [key]
            
    if failed_to_plot_keys != []:
        print("Warning: failed to plot for keys: "+', '.join(failed_to_plot_keys))
    legend(badness=legend_badness)



#### colorbars and colors ####

def colorbar(im=None, ax=None, loc="right", size="5%", pad=0.05, label=None,
             sca=True, clim=(None, None), discrete=False, Nticks_max=10, step=1,
             grid=True, grid_params=dict(grid=True), **kwargs):
    """draws vertical colorbar with decent size and positioning to the right of data.
    
    Parameters
    ----------
    im : None or matplotlib.image.AxesImage object. Default: None
        If None, set to current image.
    ax : None or Axes object. Default: None
        If None, set to current axes.
    loc : string. Default: "right"
        location of colorbar. e.g. "right", "bottom".
        If "bottom", may want to also input orientation="horizontal".
    size : string. Default: "5%"
        width compared to image. e.g. "5%" means cbar_xaxis is 5% of img_xaxis if loc="right".
    pad : float. Default: 0.05
        padding (in inches?) between plot and colorbar.
    label : None, or string. Default: None
        if passed, will use defaults for pqol.clabel() to label colorbar.
    sca : bool. Default: True
        whether to set current axes back to image after creating colorbar.
        (plt commands affect current axes by default. After calling colorbar():
        with sca= True, calling plt.title("X") puts X as title for image;
        with sca=False, calling plt.title("X") puts X as title for colorbar.)
    clim : (vmin, vmax). Default: (None, None)
        limits for colorbar.
    discrete : bool or positive number. Default: False
        whether to display colorbar as if cmap for im is discrete.
        Expands colorbar vmin&vmax to attempt to align ticks on centers of colors.
    Nticks_max : positive integer. Default: 10
        max number of ticks for colorbar.
        currently only implemented for discrete colorbars.
    step : step per color
    grid : if not True, overwrites grid in grid_params.
    grid_params: dict().
        grid : None -> no grid, True -> grid based on cmap.N (useful for discrete plots)
            or number -> number of grid boxes to draw in colorbar.
        unpacked in pqol.grid_sized() function as grid_sized(**grid_params())
    **kwargs go to plt.colorbar()
    """
    ax = ax if ax is not None else plt.gca()
    im = im if im is not None else plt.gci()
    
    ticks = kwargs.pop('ticks', None)
    if (discrete) and (ticks is None):
        ticks = _discrete_im_ticks(Nticks_max, im=im, lim=clim, step=step)
        
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    
    cbar = plt.colorbar(im, cax=cax, ticks=ticks, **kwargs)
    plt.clim(clim)
    if discrete:
        plt.clim(_discrete_clim(im))
        if grid is True: grid = grid_params.pop('grid', True)
        if grid is not None and grid is not False:
            grid = im.cmap.N if grid is True else grid
            grid_sized((grid, 1), **grid_params)
    if label is not None: clabel(label)
    if sca: plt.sca(ax) #sets current axes back to image instead of colorbar.
    return cbar
 
def clabel(label, ax=None, rotation= -90, va='baseline', **kwargs):
    """labels active Axes as if it was a vertical colorbar to the right of data."""
    ax = ax if ax is not None else plt.gca()
    return ax.set_ylabel(label, rotation=rotation, va=va, **kwargs)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map.
    
    base_cmap can be: None (-> default cmap);
    string of a valid cmap (e.g. 'Blues'. use pqol.colormaps() to see options.);
    or a cmap object.
    Adapted from https://gist.github.com/jakevdp/91077b0cae40f8f8244a  
    
    Examples
    --------
    #Try this:
    cmap=pqol.discrete_cmap(16, 'tab20')
    plt.imshow(np.arange(16).reshape(4,4), cmap=cmap)
    pqol.colorbar(discrete=True)
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def _discrete_clim(im=None, **kwargs):
    """Determine best clim for aligning tick values on colorbar for discrete im.
    
    **kwargs go to _discrete_im_info()
    """
    return _discrete_im_info(im=im, **kwargs)['clim']

def _discrete_im_info(im=None, lim=(None,None), N=None, step=1):
    """Info about image for discrete colormap.
    
    Returns a dict containing:
        N  = number of colors in map (==cmap.N - 1)
        vm = minimum value represented in plot (smallest possible tick label)
        vx = maximum value represented in plot (largest  possible tick label)
        clim = limits for colorbar for proper tick aligments
        step = step per color
    
    lim overwrites vm and vx, if lim is not None.
    N  overwrites cmap.N -1, if N is not None.
    im is used in place of plt.gci(), if im is not None.
    """
    if lim is None or N is None:
        im = im if im is not None else plt.gci()
        N  = N  if N  is not None else im.cmap.N - 1
    vm = lim[0] if lim[0] is not None else im.norm.vmin
    vx = lim[1] if lim[1] is not None else im.norm.vmax
    margin = (vx - vm)/N
    clim = (vm - margin/2, vx + margin/2)
    return dict(N=N, vm=vm, vx=vx, clim=clim, step=step)

def _discrete_im_ticks(Nticks_max=10, info=None, **kwargs):
    """Ticks to use for discrete imshow.
    
    Nticks_max limits max number of ticks; will not use more than this.
    info is _discrete_im_info(), or None to get info for current image.
    **kwargs go to _discrete_im_info()
    """
    info = info if info is not None else _discrete_im_info(**kwargs)
    N, vm, vx, step = info['N'], info['vm'], info['vx'], info['step']
    cstep = step * (N // Nticks_max + 1)
    return np.arange(vm, vx+cstep, cstep) 
    
    
def discrete_imshow(data, step=1, base_cmap=None, do_colorbar=False,
                    colorbar_params=dict(), cgrid_params=dict(), **kwargs):
    """imshow of data with discrete colormap generated automatically.
    
    To add a well-formatted discrete colorbar, use do_colorbar=True
    
    step is step between discrete values; it is 1 by default.
    base_cmap is used by discrete_cmap; see documentation there for allowed values.
    if do_colorbar is True, adds a colorbar.
    colorbar_params is unpacked in colorbar. (as colorbar(**colorbar_params))
    cgrid_params is passed to colorbar as colorbar(grid_params=cgrid_params).
    **kwargs go to imshow.
    
    returns image (=plt.imshow(...))
    """
    N = np.max(data) - np.min(data)
    cmap = discrete_cmap(N//step + 1, base_cmap=base_cmap) #integer division
    im = plt.imshow(data, cmap=cmap, **kwargs)
    ax = plt.gca()
    if do_colorbar:
        colorbar(discrete=True, step=step,
                 grid_params=cgrid_params, **colorbar_params)
    plt.sca(ax) #sets current axes to image instead of colorbar.
    return im

def Nth_color(N, cmap=None, n_discrete=None):
    """returns the Nth color in the default color cycle, or cmap if passed.
    
    N counts up from 0.
    N may be an integer or list of integers.
    if n_discrete is entered, uses cmap=discrete_cmap(n_discrete, cmap).
    if cmap is entered without n_discrete, cmap must be a colormap object.
    
    Examples
    --------
    #Nth_color(1) is orange; the second color in the default color cycle.
    #Try this:
    for i in range(12):
        plt.plot(i + np.arange(5), color=pqol.Nth_color(i, 'plasma', 10))
    """
    if cmap is None:
        colors = [x['color'] for x in list(plt.rcParams['axes.prop_cycle'])]
        return np.array(colors)[np.mod(N,len(colors))]
    elif n_discrete is None:
        return cmap(np.mod(N, cmap.N))
    else:
        return discrete_cmap(n_discrete, cmap)(np.mod(N,n_discrete))
    

#### field of view ####

## imshow field of view ##
    
def extend(center, size, shape=None):
    """returns numerical values for extent of box with same aspect ratio as data.
    
    size is number of pixels in y direction.
    shape is shape of data; data.shape.
    
    Examples
    --------
    >>> pqol.extend((5,10), 4)
    [3.0, 7.0, 8.0, 12.0]
    """
    shape = [1.,1.] if shape is None else shape
    scale = shape[1]/shape[0]
    xsize = size*scale
    ysize = size
    x_L = center[0] - xsize/2.
    y_B = center[1] - size/2.
    return [x_L, x_L + xsize, y_B, y_B + ysize]

def zoom(center, size, shape=None, ax=None):
    """Zooms into region centered on center with size size.
    
    inputs should be in axes (data) coordinates.
    Example: zoom((5, 10), 4) #will show region [3,7] x [8,12].
    """
    ax = ax if ax is not None else plt.gca()
    extent = extend(center, size, shape=shape)
    ax.set_xlim([extent[0],extent[1]])
    ax.set_ylim([extent[3],extent[2]])

def zoomregion(xm, xx, ym, yx, ax=None):
    """Zooms into region [xm,xx] [ym,yx].
    
    inputs should be in axes (data) coordinates.
    Example: zoomregion(3,7,8,12) #will show region [3,7] x [8,12].
    """
    ax = ax if ax is not None else plt.gca()
    ax.set_xlim([xm,xx])
    ax.set_ylim([ym,yx])
    
    
## plt.plot / plt.scatter field of view ##

def do_xlim(ylim=True, data=None, ax=None, margin=XYLIM_MARGIN):
    """sets xlim based on ylim. Also returns calculated xlim values.
    
    Parameters
    ----------
    ylim : True or [ymin, ymax]
        ylim of plot, or True to read it from current plot (or from ax if ax is not None).
    data : None or list like [[xarray1, yarray1], ..., [xarrayN, yarrayN]]
        data of plot, or None to read it from current plot (or from ax if ax is not None).
    ax   : None or matplotlib.axes object
        axes of plot, or None to just read from current plot.
    margin : float
        percent margin to show beyond the optimal/tight xlim. 
    """
    ax = ax if ax is not None else plt.gca()
    xlim = _find_xlim(ylim=ylim, data=data, ax=ax, margin=margin)
    ax.set_xlim(xlim)
    return xlim

def do_ylim(xlim=True, data=None, ax=None, margin=XYLIM_MARGIN):
    """sets ylim based on xlim. Also returns calculated ylim values.
    
    Parameters
    ----------
    xlim : True or [xmin, xmax]
        ylim of plot, or True to read it from current plot (or from ax if ax is not None).
    data : None or list like [[xarray1, yarray1], ..., [xarrayN, yarrayN]]
        data of plot, or None to read it from current plot (or from ax if ax is not None).
    ax   : None or matplotlib.axes object
        axes of plot, or None to just read from current plot.
    margin : float
        percent margin to show beyond the optimal/tight ylim. 
    """
    ax = ax if ax is not None else plt.gca()
    ylim = _find_ylim(xlim=xlim, data=data, ax=ax, margin=margin)
    ax.set_ylim(ylim)
    return ylim

    
def _find_xylim(xlim=None, ylim=None, data=None, ax=None, margin=XYLIM_MARGIN):        
    """returns optimal x(or y)lim based on y(or x)lim & data, with margin at edges.
    
    returns whichever lim is set to None. e.g. if xlim is None, returns xlim.
    
    if data is None, pulls data from ax.
    if xlim is True, pulls xlim info from ax.
    if ylim is True, pulls ylim info from ax.
    if ax   is None, sets ax to current plot.
    margin represents percent of y-range to extend past ylim.
    """
    if   xlim is     None and ylim is     None:
        print("Error in _query_xylim. Either xlim or ylim must not be None.")
        return None
    elif xlim is not None and ylim is not None:
        print("Error in _query_xylim. Either xlim or ylim must be None.")
        return None
    #else:
    ax   = ax   if ax   is not None else plt.gca()
    data = data if data is not None else get_data(ax, combine=True)
    xlim = xlim if xlim is not True else ax.get_xlim()
    ylim = ylim if ylim is not True else ax.get_ylim()
    x    = data[0]
    y    = data[1]
    if ylim is None:
        klim = xlim #known lim = xlim
        k, u = x, y #data for (known lim, unknown lim) = (x, y)
    else: #xlim is None:
        klim = ylim #known lim = ylim
        k, u = y, x #data for (known lim, unknown lim) = (y, x)
    ik   = (k >= klim[0]) & (k <= klim[1])
    ulim = [u[ik].min(), u[ik].max()]
    ulim = ulim + (margin * (ulim[1] - ulim[0]) * np.array([-1, 1]))
    return ulim

def _find_xlim(ylim=True, data=None, ax=None, margin=XYLIM_MARGIN):
    """returns xlim based on ylim. see _find_xylim for further documentation."""
    return _find_xylim(xlim=None, ylim=ylim, data=data, ax=ax, margin=margin)

def _find_ylim(xlim=True, data=None, ax=None, margin=XYLIM_MARGIN):
    """returns ylim based on xlim. see _find_xylim for further documentation."""
    return _find_xylim(xlim=xlim, ylim=None, data=data, ax=ax, margin=margin)
    
 
#### data overlap/density in plotspace ####
    
## get data from plot ##

def get_data(ax=None, combine=False):
    """gets data of anything plotted by plt.plot & plt.scatter, on ax.
    
    if ax is None, defaults to current axes (i.e. active plot).
    
    Returns:
    --------
    If combine is False:
        list of [xdata, ydata] arrays (potentially masked).
        .plot arrays will be listed first, followed by .scatter masked arrays.
    If combine is True:
        [all xdata (from all the plots), all corresponding ydata].
        result[0][i] will be a single xdata point, at least one of the
        plots will contain the point (result[0][i], result[1][i]),
        and every point of xdata from every plot on ax will be in result[0].
        Specifically, with data = get_data(ax, False), get_data(ax, True) will
        return [[x for l in data for x in l[0]], [y for l in data for y in l[1]]].
    (for static return formatting, not based on output, use _get_alldata.)
    """
    d = _get_alldata(ax)
    if not combine:
        return d["plot"] + d["scatter"]
    elif combine:
        data = d["plot"] + d["scatter"]
        x    = np.array([x for l in data for x in l[0]])
        y    = np.array([y for l in data for y in l[1]])
        return [x, y]
        
def _get_alldata(ax=None):
    """gets data of anything plotted by plt.plot & plt.scatter, on ax.
    if ax is None, defaults to current axes (i.e. active plot).
    returns dict with keys "plot" and "scatter";
        dict["plot"]    == list of [xdata, ydata]        arrays from .plot
        dict["scatter"] == list of [xdata, ydata] masked arrays from .scatter
    """
    ax = ax if ax is not None else plt.gca()
    return dict(
            plot    = _get_plotdata(ax),
            scatter = _get_scatterdata(ax))
    
def _get_plotdata(ax=None):
    """gets data of anything plotted by plt.plot, on ax.
    if ax is None, defaults to current axes (i.e. active plot).
    returns list of [xdata, ydata] arrays.
    """
    ax = ax if ax is not None else plt.gca()
    return [np.array(l.get_data()) for l in ax.lines]

def _get_scatterdata(ax=None):
    """gets data of anything plotted by plt.scatter, on ax.
    if ax is None, defaults to current axes (i.e. active plot).
    returns list of [xdata, ydata] masked arrays.
    """
    ax = ax if ax is not None else plt.gca()
    return [c.get_offsets().T for c in ax.collections]
        

## overlap with data in plot ##

def total_overlap(ax=None, gridsize=DEFAULT_GRIDSIZE, text_weight=1,
                  kernel_mode=True, **kernel_params):
    """determines the total overlap of plotted data & text.
    
    text_weight tells how to convert text overlap into an effective data overlap.
    The formula is:
        effective_text_overlap = text_weight * text_overlap * max(data_overlap)
    """
    
    overlap = data_overlap(ax=ax, gridsize=gridsize)
    
    if kernel_mode: overlap = _apply_kernel(overlap, **kernel_params)
    t_overlap = text_overlap(ax=ax, gridsize=gridsize)
    effective_overlap = overlap + text_weight * t_overlap * np.max(overlap)

    return effective_overlap

def text_overlap(ax=None, gridsize=DEFAULT_GRIDSIZE):
    """Determines fraction of each box in a grid which overlaps any text/legend.
    
    Uses ax, or plt.gca() if ax is not provided.
    gridsize=[N_rows, N_cols] number of boxes in y & x directions, evenly spaced.
    
    Returns array of shape==gridsize,
    with array[i][j] == fraction of box (i=row_num,j=col_num) which overlaps any text/legend.
    """
    ax = ax if ax is not None else plt.gca()
    texts  = get_texts(ax)
    legend = get_legend(ax)
    
    bboxes = [bbox_corners(text , output='axes') for text in texts]
    if legend is not None:
        bboxes += [bbox_corners(legend, output='axes')] #'+' means 'append'
    
    ys, xs = _grid_ax_coords(gridsize) #ys are in descending order
    
    r  = np.zeros(gridsize)
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            gridbox = [(xs[j], ys[i+1]), (xs[j+1], ys[i])]
            for bbox in bboxes:
                r[i,j]+=_boxes_overlap(bbox, gridbox)
    return r * np.product(gridsize)
        
def _boxes_overlap(A, B):
    """determines area of overlap between boxes a and b
    
    Each box should be entered as [(LLx, LLy), (URx, URy)],
    where LL = lower left corner, and UR = upper right corner.
    """
    LLA, URA = A
    LLB, URB = B
    if LLA[0] > URB[0] or LLB[0] > URA[0] or \
        LLA[1] > URB[1] or LLB[1] > URA[1]:
            return 0.
    else:
        return ( max(LLA[0], LLB[0]) - min(URA[0], URB[0]) ) * \
                ( max(LLA[1], LLB[1]) - min(URA[1], URB[1]) ) 

def data_overlap(ax=None, gridsize=DEFAULT_GRIDSIZE):
    """Determines number of data points overlapping each box in a grid.
    
    Uses ax, or plt.gca() if ax is not provided.
    gridsize=[N_rows, N_cols] number of boxes in y & x directions, evenly spaced.
    
    Returns array of shape==gridsize,
    with array[i][j] == number of data points in box (i=row_num,j=col_num)
    
    Examples
    --------
    #Try this:
    x=np.arange(10)
    plt.scatter(x, x,              marker='x', s=500, color='red')
    plt.scatter(x, ((x - 4)/2)**2, marker='+', s=500, color='black')
    overlap = pqol.data_overlap(gridsize=(5,3))
    im = plt.imshow(overlap,
                    extent=[*plt.gca().get_xlim(), *plt.gca().get_ylim()],
                    cmap=pqol.discrete_cmap(5, 'viridis'));
    #   ^^extent parameter is necessary to make imshow align with other plots
    pqol.colorbar(discrete=True)
    """ 
    (xdata, ydata) = get_data(ax, combine=True)
    xaxdata = _xcoords_data_to_ax(xdata, ax)
    yaxdata = _ycoords_data_to_ax(ydata, ax)
    
    ys, xs = _grid_ax_coords(gridsize)
    in_ybox = [[]]*gridsize[0]
    in_xbox = [[]]*gridsize[1]
    for i in range(gridsize[0]):
        in_ybox[i] = (ys[i] > yaxdata) & (yaxdata > ys[i+1])
    for j in range(gridsize[1]):
        in_xbox[j] = (xs[j] < xaxdata) & (xaxdata < xs[j+1])
    
    r  = np.zeros(gridsize)
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            r[i][j] = np.sum(in_xbox[j] & in_ybox[i])
    
    return r

def _grid_ax_coords(gridsize, origin="upper"):
    """returns ax coords of gridpoints for gridsize=[N_rows, N_cols].
    
    Returns [yvals, xvals] which represent intersections of gridlines;
    thus len(yi)=N_rows+1 and len(xi)=N_cols+1.
    If origin="upper", the box numbering is assumed to begin at the top left.
    """
    yl, xl = gridsize
    return [np.arange(yl + 1)[::-1]/yl, np.arange(xl + 1)/xl]

def test_overlap(box, overlap=None, gridsize=DEFAULT_GRIDSIZE, box_spec='axes',
                 **t_o_kwargs):
    """determines how much <box> will overlap with stuff on plot.
    <box> is in axes coords by default. Other options: 'data'.
    
    <box> must be in the format [(LLx, LLy), (URx, URy)],
    where LL is lower left corner, and UR is upper right corner.
    
    if overlap is None, calculates overlap using total_overlap function.
    **t_o_kwargs go to total_overlap()
    
    returns a single number - the weighted sum of overlap values in the box.
    if a gridbox partially overlaps <box>, the overlap in that gridbox will
    be weighted by the fraction of gridbox in <box>.
    """
    if overlap is None:
        overlap = total_overlap(gridsize=gridsize, **t_o_kwargs)
    else:
        gridsize=overlap.shape
        
    box = box if box_spec=='axes' else _xycoords_data_to_ax(box)
        
    ys, xs = _grid_ax_coords(gridsize) #ys are in descending order
    r  = np.zeros(gridsize)
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            gridbox = [(xs[j], ys[i+1]), (xs[j+1], ys[i])]
            r[i,j] += _boxes_overlap(box, gridbox) * overlap[i,j]
    #_boxes_overlap gives an area. multiply by number of boxes to get fraction.
    return np.sum(r * np.product(gridsize))
    

## image blurring ##

def _apply_kernel(data, edge_method='drop', shape=(5,5), **kernel_kwargs):
    '''applies kernel to data. kwargs go to _make_kernel.
    
    shape is shape of kernel.
    
    edge_methods
    ------------
    How to handle edges (when center pixel is close to edge of data).
    Caution: will get confused if kernel shape is bigger than data shape.
    'drop'    ->
        ignore the non-existing pixels; re-weight kernel with only the existing ones.
    '''
    base_kernel = _make_kernel(shape=shape, **kernel_kwargs)
    if base_kernel is None: return None
    
    output = np.zeros(data.shape)
    xl, yl = data.shape
    
    margin = [(shape[0]-1)//2, (shape[1]-1)//2] #based on kernel shape
    
    def crosses_edge(ii, l):
        '''whether ii is a slice which crosses an edge (at 0 or l).
        returns 1 if close to 0, 2 if close to l, 0 if not within margin.'''
        return (1) if ii[0] < 0 else (2) if ii[1] > l else (0)
    
    #LOOP THROUGH PIXELS
    for i in range(xl):
        ikernel = base_kernel
        xii = np.array([i - margin[0], i + margin[0] + 1])
        cex = crosses_edge(xii, xl)
        if cex:
            if edge_method=='drop':
                if cex==1: #close to i==0;  xii[0]<0
                    xadd = np.array([- xii[0],      0      ])
                    xii  = xii + xadd
                else:      #close to i==xl; xii[1]>xl
                    xadd = np.array([    0   , xl - xii[1] ])
                    xii  = xii + xadd
                ikernel = ikernel[ slice(*(xadd + [0,shape[0]])), slice(None) ]
                ikernel = ikernel / np.sum(ikernel)
        for j in range(yl):
            kernel = ikernel
            yii = np.array([j - margin[1], j + margin[1] + 1])
            cey = crosses_edge(yii, yl)
            if cey:
                if edge_method=='drop':
                    if cey==1: #close to j==0;  yii[0]<0
                        yadd = np.array([- yii[0],      0      ])
                        yii  = yii + yadd
                    else:      #close to j==yl; yii[1]>yl
                        yadd = np.array([    0   , yl - yii[1] ])
                        yii  = yii + yadd
                    kernel = kernel[ slice(None), slice(*(yadd + [0,shape[1]])) ]
                    kernel = kernel / np.sum(kernel)
            #print(i, j, kernel, xii, yii, data[slice(*xii), slice(*yii)])
            output[i, j] = np.sum(kernel * data[slice(*xii), slice(*yii)])
    return output

def _make_kernel(shape=(5,5), f=None, sigma=0.5, A=None, **fkwargs):
    '''makes a 2d kernel with shape shape according to function f.
    
    Evaluates f in the box [-1,1]x[-1,1]. 
    
    if no f is entered, makes a gaussian kernel with sigx=sigy=sigma,
    centered at middle of shape.
    if sigx or sigy appear in fkwargs, they overwrite sigma for that direction.
    
    A multiplies the overall value of the kernel.
    If A is None, instead ensure that the sum of the kernel pixels is 1.
    Additional kwargs are passed to f.
    '''
    if shape[0]%2==0 or shape[1]%2==0:
        print("ERROR: kernel shape must be odd for centering purposes.")
        return None
    x = np.linspace(-1,1,shape[0])
    y = np.linspace(-1,1,shape[1])
    x, y = np.meshgrid(x, y, sparse=True)
    
    if f is None:
        sigx = fkwargs.pop('sigx', sigma)
        sigy = fkwargs.pop('sigy', sigma)
        def gaussian_func(x, y, sigx=sigx, sigy=sigy):
            return np.exp(-(x**2/(2*sigx) + y**2/(2*sigy)))
        f = gaussian_func
    
    kernel = f(x, y)
    A = A if A is not None else 1/kernel.sum()
    
    return A * kernel

#### annotation ####

def text(s, ax_xy=None, iters=50, ax=None, gridsize=DEFAULT_GRIDSIZE,
         overlap=None, overlap_params=dict(), 
         allow_external=False, external_margin=-0.01, **kwargs):
    """puts textbox with text s.
    
    By default, puts where pqol thinks is best, based on data in plot.
    
    If ax_xy is passed, instead places text at ax coordinates ax_xy= (ax_x, ax_y).
    e.g. ax_xy = (0.7, 0.2) places text 70% across from left, & 20% up from bottom.
    If ax_xy is not passed, picks location based on data in plot.
    increase badness value to use next-to-best locations.
    For coordinates in terms of data, use plt.text().
    **kwargs go to plt.text()
    
    Example
    -------
    #Try this! pqol will ensure the textboxes do not overlap, automagically!
    plt.plot(-np.arange(10), label='line')
    pqol.legend()
    pqol.text('hello')
    pqol.text('look, a wild textbox!')
    pqol.text('this is a very very very very large one')
    pqol.text('this is another very large box')
    pqol.text('this box is less large')
    pqol.text('doing a test')
    pqol.text('one last test :)')
    
    #For fun, you can look at the overlap matrix that pqol used:
    overlap = pqol.total_overlap()
    pqol.imshow_overplot(overlap)
    pqol.colorbar()
    """
    default_bbox = dict(facecolor='none')
    default_ha   = 'center'
    default_va   = 'center'
    
    bbox = kwargs.pop('bbox', default_bbox)
    ha = kwargs.pop('verticalalignment', None)
    ha = ha if ha is not None else kwargs.pop('ha', default_ha)
    va = kwargs.pop('verticalalignment', None)
    va = va if va is not None else kwargs.pop('va', default_va)
    
    if ax_xy is not None:
        x, y = ax_xy
        x = _xcoords_ax_to_data(x, ax=ax)
        y = _ycoords_ax_to_data(y, ax=ax)
    else:
        if overlap is None:
            overlap = total_overlap(ax, gridsize=gridsize, **overlap_params)
    
        axlocs = locs_best(ax=ax, gridsize=gridsize, overlap=overlap)
        
        smallest_ol = np.inf
        smallest_ol_i = 0
        if iters < 0: iters = int(np.min([-1 * iters, 1]) * np.product(gridsize))
        i=-1
        while i<iters-1 or (smallest_ol is np.inf and i<np.product(gridsize)):
            i+=1
            y, x = axlocs['loc'][i] #ax_y & ax_x of lower left corner of box
            x = _xcoords_ax_to_data(x, ax=ax)
            y = _ycoords_ax_to_data(y, ax=ax)
            t = plt.text(x, y, s, bbox=bbox, ha=ha, va=va, **kwargs)
            bbc = bbox_corners(t, output='axes')
            t.remove() #prevents cluttering as we look for best possible location via trial & error.
            if not allow_external:
                outside = np.sum( (bbc > 1+external_margin) | (bbc < 0-external_margin) ) > 0
                if outside:
                    continue
            ol = test_overlap(bbc, overlap=overlap, box_spec='axes')
            if ol < smallest_ol:
                smallest_ol_i = i
                smallest_ol   = ol
        y, x   = axlocs['loc'][smallest_ol_i] #ax_y & ax_x of lower left corner of BEST box.
        x = _xcoords_ax_to_data(x, ax=ax)
        y = _ycoords_ax_to_data(y, ax=ax)
    t = plt.text(x, y, s, bbox=bbox, ha=ha, va=va, **kwargs)
    return t
     
def legend(iters=20, ax=None, gridsize=DEFAULT_GRIDSIZE, overlap=None,
           loc='center', overlap_params=dict(),
           allow_external=False, external_margin=-0.03,
           **kwargs):
    """puts a legend where pqol thinks is best, based on data in plot.
    
    iters is number of gridpoints to test.
        if negative, test iters * total_number_of_gridpoints points.
    (Tests points in ascending order of overlap with other plot elements.)
    gridsize allows for finer or coarser search.
    loc is location INSIDE best grid box.
    **kwargs go to plt.legend().
    overlap_params gets unpacked in total_overlap()
    allow_external is whether to allow part of the legend to be outside the axes area
    external_margin is what counts as external to the axes area.
        (with a margin of 0.05, 'outside' is when axes coords are <-0.05 or >1.05.)
    
    for legend location based on axes coordinates, instead,
    use plt.legend(loc=(x, y)) to place bottom left corner of legend at x,y.
    """
    if overlap is None:
        overlap = total_overlap(ax, gridsize=gridsize, **overlap_params)
    
    axlocs = locs_best(ax=ax, gridsize=gridsize, overlap=overlap)
    
    smallest_ol = np.inf
    smallest_ol_i = 0
    if iters < 0: iters = int(np.min([-1 * iters, 1]) * np.product(gridsize))
    i=-1
    while i<iters-1 or (smallest_ol is np.inf and i<np.product(gridsize)):
        i+=1
        y, x   = axlocs['loc'][i] #ax_y & ax_x of lower left corner of box
        l = plt.legend(bbox_to_anchor=(x, y, axlocs["w"], axlocs["h"]), loc=loc, **kwargs)
        bbc = bbox_corners(l, output='axes')
        if not allow_external:
            outside = np.sum( (bbc > 1+external_margin) | (bbc < 0-external_margin) ) > 0
            if outside:
                continue
        ol = test_overlap(bbc, overlap=overlap, box_spec='axes')
        if ol < smallest_ol:
            smallest_ol_i = i
            smallest_ol   = ol
    y, x   = axlocs['loc'][smallest_ol_i] #ax_y & ax_x of lower left corner of BEST box.
    l = plt.legend(bbox_to_anchor=(x, y, axlocs["w"], axlocs["h"]), loc=loc, **kwargs)
    return l

def locs_visual(ax=None, gridsize=DEFAULT_GRIDSIZE, overlap=None,
                cmap='cividis', **kwargs):
    """visual representation of emptiest locations based on overlap with data.
    
    works in kernel_mode=False. Recommended to not use during kernel_mode=True.
    
    overplots a grid of numbered boxes, numbered according to their 'badness'.
    'badness' measures overlap with data, and the numbers on the plot from this
    function agree with the badness keyword in pqol.legend() and pqol.text().
    ties are determined arbitrarily (by default np.argsort sorting of overlap).
    **kwargs go to imshow.
    
    returns result of locs_best().
    
    Examples
    --------
    #try this:
    x = np.arange(-5, 4, 0.7)
    plotstyle = dict(markersize=20, fillstyle='none')
    plt.plot(x,       x**2     , marker='^', **plotstyle)
    plt.plot(x, 8*(1+np.cos(x)), marker='o', **plotstyle)
    pqol.locs_visual()
    """
    if overlap is None: overlap = data_overlap(ax=ax, gridsize=gridsize)
    else: gridsize = overlap.shape
    ii = _locs_best_i(ax=ax, gridsize=gridsize, overlap=overlap)
    axlocs = locs_best(gridsize=gridsize, locs_best_i=ii, kernel_mode=False)
    w2, h2 = axlocs['w']/2, axlocs['h']/2
    for i in range(len(axlocs['loc'])):
        y, x = axlocs['loc'][i]
        text(str(i), (x+w2, y+h2)) #<<< badness --> text in center of gridbox.
        text("N = "+str(int(overlap[ ii[0][i],ii[1][i] ])), 
             (x+axlocs['w'], y), bbox=None, fontsize=10, va='bottom', ha='right') 
    plt.imshow(overlap,
               extent = [*plt.gca().get_xlim(), *plt.gca().get_ylim()],
               cmap   = discrete_cmap(overlap.max()+1, cmap),
               alpha  = 0.3,
               aspect = 'auto',
               **kwargs)
    ax = plt.gca()
    grid_sized(gridsize, color='black', lw=1)
    colorbar(discrete=True)
    plt.sca(ax) #sets current axes back to ax instead of colorbar.
    return axlocs

def imshow_overplot(data, **imshow_kwargs):
    '''imshows data over current axes.'''
    plt.imshow(data,
               extent=[*plt.gca().get_xlim(), *plt.gca().get_ylim()],
               aspect=imshow_kwargs.pop('aspect', 'auto'),
               **imshow_kwargs)
    
def locs_best(ax=None, gridsize=DEFAULT_GRIDSIZE, overlap=None, locs_best_i=None,
              text_weight=1, kernel_mode=True, **kernel_params):
    """returns emptiest locations, in axes coordinates, based on overlap with data.
    
    return will be a dict with keys "loc", "w", "h".
    r["loc"][i] will be the axis coords (y, x) for the lower left corner of
    the i'th emptiest gridbox. 
    (r["w"], r["h"]) will be the (width,height) in axes coords of a gridbox.
    
    if overlap is not None, ignores gridsize & ax.
    if locs_best_i is not None, this is used instead of doing _locs_best_i().
    
    if kernel_mode is True, blurs the overlap data (gaussian kernel by default).
    kernel_params go to _apply_kernel.
    if kernel_mode is True, use gridsize kgridsize instead.
    """
    
    if locs_best_i is not None: ii = locs_best_i
    else: ii = _locs_best_i(ax=ax, gridsize=gridsize, overlap=overlap,
                            kernel_mode=kernel_mode, kernel_params=kernel_params)
    
    gridsize = gridsize if overlap is None else overlap.shape
    ys, xs   = _grid_ax_coords(gridsize)
    
    axlocs = [(ys[yi+1], xs[xi]) for yi,xi in np.transpose(ii)]
    return dict(loc=axlocs, w=xs[1]-xs[0], h=ys[0]-ys[1])

def _locs_best_i(ax=None, gridsize=DEFAULT_GRIDSIZE, overlap=None,
                 text_weight=1, kernel_mode=True, **kernel_params):
    """returns empitest locations, in grid indices, based on overlap.
    
    return will be a list [yvals, xvals], with each (yvals[i], xvals[i])
    being the indices for the i'th emptiest gridbox.
    
    if overlap is passed, ignores all other parameters.
    
    if kernel_mode is True, blurs the overlap data (gaussian kernel by default).
    kernel_params go to _apply_kernel.
    if kernel_mode is True, use gridsize kgridsize instead.
    
    text_weight tells how to convert text overlap into an effective data overlap.
    The formula is:
        effective_text_overlap = text_weight * text_overlap * max(data_overlap)
    """    
    if overlap is None:
        overlap = total_overlap(ax=ax, gridsize=gridsize, text_weight=text_weight,
                                kernel_mode=kernel_mode, **kernel_params)
    else:
        gridsize = overlap.shape

    return np.unravel_index(overlap.argsort(axis=None), gridsize)    
    
def linecalc(x1x2,y1y2):
    """returns the parameters for the line through (x1,y1) and (x2,y2)"""
    x1,x2=x1x2
    y1,y2=y1y2
    m=(y1-y2)/(x1-x2)
    b=(x1*y2-x2*y1)/(x1-x2)
    return dict(m=m,b=b)
    
def plotline(xdata, m, b, xb=0, **kwargs):
    """plots the line with parameters m=slope & b=y-intercept, at xdata.
    
    xb = (x where y==b). usually 0; use xb!=0 for point-slope form of line.
    """
    yline  = np.array(m * (xdata - xb) + b)
    plt.plot(xdata, yline, **kwargs)
    return [xdata, yline]

def draw_box(center, width, ax=None, height=None):
    """draws square if height is not entered, else draws box."""
    #patches.Rectangle needs top right corner
    height = height if height is not None else width
    ax = ax if ax is not None else plt.gca()
    x_L = center[0] - width/2.
    y_B = center[1] - height/2.
    ax.add_artist(patches.Rectangle((x_L,y_B),width,height,fill=False,color='red'))
    
def labelline(xy, xyvars=('x', 'y'), s="{:} = {:.2e} * {:} + {:.2e}"):
    """returns string label using slope & intercept from linregress(xy)"""
    l = linregress(xy)
    return s.format(xyvars[1], l.slope, xyvars[0], l.intercept)

def grid_sized(gridsize, ax=None, color='black', **kwargs):
    """draws a grid of size gridsize=(Nrows,Ncols)=(Nboxes_y, Nboxes_x).
    
    **kwargs go to grid() -> vline()/hline() -> plt.axvline()/plt.axhline().
    """
    ys, xs = _grid_ax_coords(gridsize)
    ys = _ycoords_ax_to_data(ys, ax=ax)
    xs = _xcoords_ax_to_data(xs, ax=ax)
    return grid(xs[1:-1], ys[1:-1], color=color, **kwargs) #probably None-typed

def grid(x, y, ax=None, hparams=dict(), vparams=dict(), **kwargs):
    """draws a grid on plot. **kwargs go to hline & vline.
    
    x = list of x values for gridlines, in data coordinates.
    y = list of y values for gridlines, in data coordinates.
    hparams / vparams pass keywords to hline / vline.
    hparams / vparams override any conflicting kwargs.
    
    For labeling purposes,
    '*x' / '*y' will be replaced with x / y value at vline / hline.
    
    Examples
    --------
    #try this:
    xvals = np.arange(-5, 4, 0.2)
    plt.plot(xvals, 2*np.sin(xvals))
    pqol.grid([-3.1415, -0.5, 2.9],[-2, -1.5, 0 ,2.0],
              vparams=dict(text='value=*x', textloc=0.9, textdirection='down',
                           textside='right', textparams=dict(fontweight='bold')),
              hparams=dict(color='green'),
              linestyle='-.', color='black')
    """
    vp = dict(); vp.update(kwargs); vp.update(vparams)
    hp = dict(); hp.update(kwargs); hp.update(hparams)
    if 'text' in vp.keys(): vtext = vp['text']
    if 'text' in hp.keys(): htext = hp['text']
    update_text_v = 'text' in vp.keys() and '*x' in vp['text']
    update_text_h = 'text' in hp.keys() and '*y' in hp['text']
    for xval in x:
        if update_text_v: vp['text']=vtext.replace('*x',str(xval))
        vline(xval, **vp)
    for yval in y:
        if update_text_h: hp['text']=htext.replace('*y',str(yval))
        hline(yval, **hp)
    
def hline(y, text=None, textparams=dict(), textloc=0.5, textanchor="start",
          textside="top", textmargin=TEXTBOX_MARGIN,
          xmin=0, xmax=1, yspec="data", xspec="axes", textspec="axes",
          ax=None, forceplot=False, **kwargs):
    """Add a horizontal line across the plot, and label it if 'text' is entered.
    
    Use 'text' parameter to add text along line.
    additional **kwargs are passed to plt.axhline.
    "axes" coordinates refers to coordinates of the Axes.
    (0,0) is the bottom left of the axes,
    (0,1) is the top left, and (1,1) is the top right.

    Parameters
    ----------
    y : scalar
        y position of the horizontal line. In data coordinates by default.
    text : string or None. Default: None
        text to label line on the plot, or None.
    textparams : dict. Default: {}
        pass parameters to ax.text() function here.
        e.g. textparams=dict(fontsize=10) will change the fontsize to 10.
    textloc : scalar. Default: 0.5
        location of text anchor along line, in axes coordinates by default.
    textanchor : "left", "center", or "right". Default: "center"
        which part of text is anchored at textloc.
        "start"  -> the text will begin at textloc.
        "center" -> the text will be centered on textloc.
        "end"    -> the text will end at textloc.
    textside : "top", "center", or "bottom". Default: "top"
        side of line to put text on.
        "top"    -> the text will be above the line.
        "center" -> the text will be inline. (with white bbox covering line.)
        "bottom" -> the text will be below the line.
    textmargin : scalar. Default: pqol.TEXTBOX_MARGIN (==0.002 by default)
        fraction of axes length to push text away from line.
        has no effect if textside=="center".
    xmin : scalar. Default: 0
        leftward extent of line, in axes coordinates by default.
    xmax : scalar. Default: 1
        rightward extent of line, in axes coordinates by default.
    yspec : "data" or "axes". Default: "data"
        specification coordinate system for 'y'.
    xspec : "data" or "axes". Default: "axes"
        specification of coordinate system for 'xmin' and 'xmax'.
    textspec : data" or "axes". Default: "axes"
        specification of coordinate system for 'textloc'.
    forceplot : bool. Default: False
        if false: hline will not draw lines beyond 2x axis limits away from axis.
    additional **kwargs get passed to ax.axhline
        
    Examples
    --------
    #Try this
    plt.plot(range(10))
    pqol.hline(3, "hello, world!", textloc=7, textanchor="start", textspec="data")
    pqol.vline(7, color="black")
    """
    ax = ax if ax is not None else plt.gca()
    y_d    = y if yspec=="data" else \
                (_ycoords_ax_to_data(y, ax=ax) \
                     if yspec=="axes" else None) #y in data coords
    (xmin_a, xmax_a) = (xmin, xmax) if xspec=="axes" else \
                (_xcoords_data_to_ax((xmin, xmax), ax=ax) \
                     if xspec=="data" else (None, None)) #xmin & xmax in ax coords
    
    y_a = _ycoords_data_to_ax(y_d, ax=ax)
    if (not forceplot) and abs(y_a)>2:
        print("Refusing to draw hline at y=",y_a,"in axes coords.\n" +\
              "    Use hline(forceplot=True) to draw this line anyway.")
        return #exits hline entirely.
    
    ax.axhline(y_d, xmin_a, xmax_a, **kwargs)
    
    if text is not None:
        va = "bottom" if textside=="top"   else ( \
                "center" if textside=="center"   else ( \
                "top"    if textside=="bottom"   else None)) #vertical alignment
        ha = "left" if textanchor=="start" else ( \
                "center" if textanchor=="center" else ( \
                "right"  if textanchor=="end"    else None)) #horizontal alignment
        bbox = textparams.pop('bbox', \
                 None if va != "center" else dict(fc="white", lw=0)) #default if bbox not in textparams
        
        marginal_text_shift = _margin(textmargin, "y", ax)
        marginal_text_shift *= -1 if textside=="bottom" else( \
                                0 if textside=="center" else( \
                               +1 if textside=="top" else None))
        textx_d = textloc if textspec=="data" else \
                    _xcoords_ax_to_data(textloc, ax=ax) #x for text anchor in data coords
        
        txt=ax.text(textx_d, y_d + marginal_text_shift, text, 
                    **textparams, bbox=bbox, va=va, ha=ha)
        return txt

def vline(x, text=None, textparams=dict(), textloc=0.5, textanchor="start",
          textside="left", textdirection="up", textmargin=TEXTBOX_MARGIN,
          ymin=0, ymax=1, xspec="data", yspec="axes", textspec="axes",
          ax=None, forceplot=False, **kwargs):
    """Add a vertical line across the plot, and label it if 'text' is entered.
    
    Use 'text' parameter to add text along line.
    Text will read from bottom to top by default - use 'textdirection' to change this.
    additional **kwargs are passed to plt.axhline.
    "axes" coordinates refers to coordinates of the Axes.
    (0,0) is the bottom left of the axes,
    (0,1) is the top left, and (1,1) is the top right.

    Parameters
    ----------
    x : scalar
        x position of the vertical line. In data coordinates by default.
    text : string or None. Default: None
        text to label line on the plot, or None.
    textparams : dict. Default: {}
        pass parameters to ax.text() function here.
        e.g. textparams=dict(fontsize=10) will change the fontsize to 10.
    textloc : scalar. Default: 0.5
        location of text anchor along line, in axes coordinates by default.
    textanchor : "left", "center", or "right". Default: "center"
        which part of text is anchored at textloc.
        "start"  -> the text will begin at textloc.
        "center" -> the text will be centered on textloc.
        "end"    -> the text will end at textloc.
    textside : "left", "center", or "right". Default: "left"
        side of line to put text on.
        "left"   -> the text will be to the left of the line.
        "center" -> the text will be inline. (with white bbox covering line.)
        "right"  -> the text will be to the right of the line.
    textdirection : "up" or "down". Default: "up"
        direction text should read. ("up" reads bottom to top.)
    textmargin : scalar. Default: pqol.TEXTBOX_MARGIN (==0.002 by default)
        fraction of axes length to push text away from line.
        has no effect if textside=="center".
    textrotation : 
    ymin : scalar. Default: 0
        leftward extent of line, in axes coordinates by default.
    ymax : scalar. Default: 1
        rightward extent of line, in axes coordinates by default.
    xspec : "data" or "axes". Default: "data"
        specification coordinate system for 'x'.
    yspec : "data" or "axes". Default: "axes"
        specification of coordinate system for 'ymin' and 'ymax'.
    textspec : data" or "axes". Default: "axes"
        specification of coordinate system for 'textloc'.
    forceplot : bool. Default: False
        if false: vline will not draw lines beyond 2x axis limits away from axis.
    additional **kwargs get passed to ax.axvline
        
    Examples
    --------
    #Try this
    plt.plot(range(10))
    pqol.vline(3, ">LeftUp") #default textside, default textdirection.
    pqol.vline(3, ">LeftDown",  textdirection='down') #default textside.
    pqol.vline(3, ">RightUp",   textside='right') #default textdirection.
    pqol.vline(3, ">RightDown", textside='right', textdirection='down')
    
    #TODO: investigate why the <SameSide> Up & Down texts dont line up horizontally.
    
    """
    ax = ax if ax is not None else plt.gca()
    x_d    = x if xspec=="data" else \
                (_xcoords_ax_to_data(x, ax=ax) \
                     if xspec=="axes" else None) #x in data coords
    (ymin_a, ymax_a) = (ymin, ymax) if yspec=="axes" else \
                (_ycoords_data_to_ax((ymin, ymax), ax=ax) \
                     if yspec=="data" else (None, None)) #ymin & ymax in ax coords
    
    x_a = _xcoords_data_to_ax(x_d, ax=ax)
    if (not forceplot) and abs(x_a)>2:
        print("Refusing to draw vline at x=",x_a,"in axes coords.\n" +\
              "    Use vline(forceplot=True) to draw this line anyway.")
        return #exits hline entirely.
    
    ax.axvline(x_d, ymin_a, ymax_a, **kwargs)
    
    if text is not None:
        ha = "left"   if textside=="right"   else ( \
                "center" if textside=="center"   else( \
                "right"  if textside=="left"     else None)) #horizontal alignement
        va = "bottom" if textanchor=="start" else ( \
                "center" if textanchor=="center" else( \
                "top"    if textanchor=="end"    else None)) #vertical alignment
        if textdirection=="up":
            rotation = textparams.pop('rotation', 90) #default rotation to 90 deg
            #va = va #it does not need to change.
        elif textdirection=="down":
            rotation = textparams.pop('rotation', 270) #default rotation to 270 deg
            if   va=="bottom": va = "top"
            elif va=="top":    va = "bottom"
        else:
            raise Exception("textdirection must be 'up' or 'down', only.")
        bbox = textparams.pop('bbox', \
                 None if ha != "center" else dict(fc="white", lw=0)) #default if bbox not in textparams
        marginal_text_shift = _margin(textmargin, "x", ax)
        marginal_text_shift *= -1 if textside=="left"   else( \
                                0 if textside=="center" else( \
                               +1 if textside=="right"  else None))
        texty_d = textloc if textspec=="data" else \
                    _ycoords_ax_to_data(textloc, ax=ax) #y for text anchor in data coords
        
        txt=ax.text(x_d + marginal_text_shift, texty_d, text, rotation=rotation,
                    **textparams, bbox=bbox, va=va, ha=ha)
        return txt

## Reference existing annotation ##

def bbox_corners(obj, output='axes'):
    """returns corners of text or legend object in output coordinates.
    let C1 = lower left corner, C2 = upper right corner.
    This function returns [[C1x, C1y], [C2x, C2y]].
    
    By default, uses 'axes' coordinates. Other options are: 'data'.
    
    other objects may be entered, provided they have a get_window_extext attribute.
    """
    bb = obj.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
    trans = plt.gca().transAxes if output=='axes' \
        else plt.gca().transData if output=='data' \
        else None
    return trans.inverted().transform(bb)

def get_texts(ax=None):
    """returns list of text objects placed on axes (current axes, by default)."""
    ax = ax if ax is not None else plt.gca()
    return ax.texts

def get_legend(ax=None):
    """returns legend object for axes (current axes, by default)."""
    ax = ax if ax is not None else plt.gca()
    return ax.legend_
    

## convert between data and axes coordinates ##

def _xy_ax_data_convert(coords, axis="x", convertto="data", ax=None, lim=None):
    """converts coords between ax and data coord systems.
    
    coords : scalar or array.
    axis : "x" or "y".
    convertto : "data" or "ax".
    ax : Axes object, or None to use active plot.
    lim : None, or [min, max] -> ignore plot; convert as if ax had lim=[min, max].
    If lim is not None, 'axis' and 'ax' are ignored entirely.
    """
    if lim is None:
        ax  = ax if ax is not None else plt.gca()
        lim = ax.get_xlim() if axis=="x" else ax.get_ylim()
        
    if convertto=="data":
        return lim[0] + coords * np.array( lim[1] - lim[0] ) #np.array for typecasting only.
    else:
        return (coords - lim[0]) / np.array ( lim[1] - lim[0] ) #np.array for typecasting only.
        
def _xcoords_ax_to_data(  x_axcoords  , ax=None, xlim=None):
    """Converts x-axis ax coords to x-axis data coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(x_axcoords, axis="x", convertto="data", ax=ax, lim=xlim)

def _ycoords_ax_to_data(  y_axcoords  , ax=None, ylim=None):
    """Converts y-axis ax coords to y-axis data coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(y_axcoords, axis="y", convertto="data", ax=ax, lim=ylim)

def _xycoords_ax_to_data( xy_axcoords , ax=None, xlim=None, ylim=None):
    """Converts xy ax coords into xy data coords. e.g. [(x1,y1),(x2,y2),(x3,y3)]."""
    x, y = np.transpose(xy_axcoords)
    return np.transpose([_xcoords_ax_to_data(x, ax=ax, xlim=xlim), 
                         _ycoords_ax_to_data(y, ax=ax, ylim=ylim)])

def _xcoords_data_to_ax( x_datacoords , ax=None, xlim=None):
    """Converts x-axis data coords to x-axis ax coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(x_datacoords, axis="x", convertto="ax", ax=ax, lim=xlim)

def _ycoords_data_to_ax( y_datacoords , ax=None, ylim=None):
    """Converts y-axis data coords to y-axis ax coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(y_datacoords, axis="y", convertto="ax", ax=ax, lim=ylim)

def _xycoords_data_to_ax(xy_datacoords, ax=None, xlim=None, ylim=None):
    """Converts xy data coords into xy ax coords. e.g. [(x1,y1),(x2,y2),(x3,y3)]."""
    x, y = np.transpose(xy_datacoords)
    return np.transpose([_xcoords_data_to_ax(x, ax=ax, xlim=xlim), 
                         _ycoords_data_to_ax(y, ax=ax, ylim=ylim)])
    
def _margin(margin, axis="x", ax=None):
    """converts margin as fraction of full data range into data value.
    
    returns magin * (Zmax - Zmin), where 'Z' == axis == 'x' or 'y'.
    
    ax is set to active plot if None is passed.
    """
    ax = ax if ax is not None else plt.gca()
    lim = ax.get_xlim() if axis=="x" else (ax.get_ylim() if axis=="y" else None)
    return margin * (lim[1] - lim[0])


#### Save Plots ####

## Save plots ##

def set_savedir(new_savedir=None, DEFAULT=DEFAULT_SAVEDIR):
    """sets savedir to the new_savedir, or default if None is passed.
    
    savedir is the default location for saved plots.
    it defaults to os.getcwd()+'/saved_plots/'.
    access via: import QOL.plots as pqol; pqol.savedir
    """
    global savedir
    global savedir_is_default
    if new_savedir is None:         #set to default.
        savedir = os.getcwd()+'/saved_plots/'
        savedir_is_default = True
    else:                           #set to inputted savedir
        savedir = new_savedir
        savedir_is_default = False
    return savedir

    #SET DEFAULT savedir HERE#
savedir_is_default = locals()['savedir_is_default'] \
                     if 'savedir_is_default' in locals().keys() \
                else True

if savedir_is_default or 'savedir' not in locals().keys():
    set_savedir()
    #back to function definitions#

def _convert_to_next_filename(name, folder=None, imin=None):
    """returns string for next filename starting with name, in folder.
    
    if folder is None, uses folder=savedir.
    
    Folder will be part of return; i.e. result contains the full path.
    If name starts with '/', ignores the 'folder' kwarg.
    Attempts to make folder if implied folder does not exist.
    Convention is to use filenames in order: [name, 1- name, 2- name, ...].
    Will not re-use earlier names if files are deleted.
    e.g. if only [name, 3- name, 4- name] exist, will return 5- name, not 1- name.
    if imin is not None, starts labeling at imin.
    """
    split = "- " #splits name and number.
    folder = folder if folder is not None else savedir
    
    if name.startswith('/'):
        i = name.rfind('/')
        folder = name[   : i] +'/'
        name   = name[i+1:  ]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    l = [N.split('.')[0].replace(name, '').replace(split, '') \
            for N in os.listdir(folder) if N.split('.')[0].endswith(name)]
    if not '' in l and imin is None:
        return folder + name #since default name does not already exist as file.
    else:
        if '' in l: l.remove('')
        x = np.max([int(N) for N in l]) if len(l)>0 else 0
        imin = imin if imin is not None else 0
        return folder + str(max(x + 1, imin)) + split + name

def savefig(fname=None, folder=None, Verbose=True, imin=None, **kwargs):
    """Saves figure via plt.savefig()
    
    if folder is None, uses folder=savedir.
    
    Default functionality is to save current active figure, to folder pqol.savedir.
    The filename defaults to "Untitled X" where X starts blank but counts up,
    ensuring that similarly-named files are not overwritten.
    See pqol._convert_to_next_filename,
    and the documentation below, for more details.
    
    Parameters
    ----------
    fname : str, or None. Default: None
        Location to save file.
        If None, save to _convert_to_next_filename(DEFAULT_SAVE_STR, folder)
        If str beginning with '/', save to _convert_to_next_filename(fname)
    folder : str. Default: savedir
        Folder in which to save file.
        Ignored if fname starts with '/'
        If True, prints location to which the plot is saved.
    **kwargs are passed to plt.savefig
    """
    fname = fname if fname is not None else DEFAULT_SAVE_STR
    saveto = _convert_to_next_filename(fname, folder=folder, imin=imin)
    bbox_inches = kwargs.pop("bbox_inches", 'tight')
    plt.savefig(saveto, bbox_inches=bbox_inches, **kwargs)
    if Verbose: print("Active plot saved to",saveto)

#### References to matplotlib built-in colors, markers, etc. ####
#copied from matplotlib docs.

## Colors ##

def colormaps(**kwargs):
    """Displays all available matplotlib colormaps. **kwargs go to imshow."""
    cmaps = [('Perceptually Uniform Sequential', [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
    
    
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    
    def plot_color_gradients(cmap_category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
        fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
        fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
    
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
    
        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name), **kwargs)
            ax.text(-.01, .5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)
    
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()
    
    
    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list)
    
    plt.show()
    
## Markers ##
    
def markers():
    
    from matplotlib.lines import Line2D
    points = np.ones(3)  # Draw 3 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})
    marker_style = dict(linestyle=':', color='0.8', markersize=10,
                        mfc="C0", mec="C0")
    
    def format_axes(ax):
        ax.margins(0)
        ax.set_axis_off()
        ax.invert_yaxis()
    
    def split_list(a_list):
        i_half = len(a_list) // 2
        return (a_list[:i_half], a_list[i_half:])
        fig, axes = plt.subplots(ncols=2)
        fig.suptitle('un-filled markers', fontsize=14)
    
    fig, axes = plt.subplots(ncols=2)
    fig.suptitle('un-filled markers', fontsize=14)
    
    # Filter out filled markers and marker settings that do nothing.
    unfilled_markers = [m for m, func in Line2D.markers.items()
                        if func != 'nothing' and m not in Line2D.filled_markers]
    
    for ax, markers in zip(axes, split_list(unfilled_markers)):
        for y, marker in enumerate(markers):
            ax.text(-0.5, y, repr(marker), **text_style)
            ax.plot(y * points, marker=marker, **marker_style)
        format_axes(ax)
    
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.7)
    plt.show()
    
    fig, axes = plt.subplots(ncols=2)
    fig.suptitle('filled markers', fontsize=14)
    for ax, markers in zip(axes, split_list(Line2D.filled_markers)):
        for y, marker in enumerate(markers):
            ax.text(-0.5, y, repr(marker), **text_style)
            ax.plot(y * points, marker=marker, **marker_style)
        format_axes(ax)

    plt.show()