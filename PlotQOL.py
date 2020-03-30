#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:52:21 2020

@author: Sevans

Useful and user-friendly/convenient codes for making matplotlib plots.
"""

#TODO: implement title for colorbar()
#TODO: implement scatter plot marker cycle
#TODO: implement different left/right yscales to put two plots on same grid.
    #Can be accomplished by just doing:
    #plt.plot(<firstplot>); plt.twinx(); plt.plot(<secondplot>)
    #However there are QOL issues with labeling, colors, etc, that could be fixed.
#TODO: implement easy annotation (textboxes etc.)
#check out illustrator
#A good example of textbox annotations can be found in answers here:
#   https://stackoverflow.com/questions/17086847/box-around-text-in-matplotlib
#TODO: use pqol.overlap() to find best location for annotations (e.g. legend).

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.stats import linregress
import os #only used for saving figures.

XYLIM_MARGIN=0.05
TEXTBOX_MARGIN=0.002
DEFAULT_SAVE_STR="Untitled"

#### set better defaults ####

def fixfonts(s=14, m=20, l=22):
    """sets better default font sizes for plots"""
    plt.rc('axes', titlesize=l)    # fontsize of the axes title
    plt.rc('figure', titlesize=l)  # fontsize of the figure title
    plt.rc('axes', labelsize=m)    # fontsize of the x and y labels
    plt.rc('font', size=m)         # controls default text sizes
    
    plt.rc('xtick', labelsize=s)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)   # fontsize of the tick labels
    plt.rc('legend', fontsize=s)   # legend fontsize

def fixfigsize(size=(8,8)):
    """sets better default figure size for plots"""
    plt.rcParams['figure.figsize'] = size
    
def set_plot_defaults():
    """sets better defauls (fonts & figsize) for plots"""
    fixfigsize()
    fixfonts()
    
set_plot_defaults() #actually sets the defaults upon loading/importing PlotQOL.py


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

def iplot(x, y=None, ss=None, i=None, plotter=plt.plot, iplotter=None, **kwargs):
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
        return plotter(x[s], y[s], **kwargs)
    elif i is not None:
        return plotter(x[i], y[i], **kwargs)
    else:
        return plotter(x, y, **kwargs)
    
def dictplot(x, y=None, yfunc=lambda y:y, xfunc=lambda x:x, keys=None, hide_keys=[],
             keys_prefix=None, hide_keys_prefix=[], prefix="", suffix="", 
             plotter=plt.plot, **kwargs):
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
    hide_keys : [], or list of strings.
        Will not show any key in hide_keys.
    keys_prefix : None, string, or list of strings.
        Only plot keys starting with keys_prefix in dict.
        If keys is not None, adds to list of keys instead
    hide_keys_prefix : [], string, or list of strings.
        Hide all keys starting with any hide_keys_prefix.
        Subtracts from list of keys to plot if hide_keys_prefix is not None.
    prefix : string
        prefix to all labels. (useful if plotting multiple dicts with same keys.)
    suffix : string
        suffix to all labels. (useful if plotting multiple dicts with same keys.)
    plotter : function
        Must accept x and y as args; label and **kwargs as kwargs.
    **kwargs are passed to plotter.
    
    Examples
    --------
    #import PlotQOL as pqol; then try the following:
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
    def set_keys(d, keys=None, keys_prefix=None, hide_keys=[], hide_keys_prefix=[]):
        do_keys = d.keys() if (keys is None and keys_prefix is None)     else \
                    []     if (keys is None and keys_prefix is not None) else \
                    keys
        if keys_prefix is not None:
            if type(keys_prefix)==str: keys_prefix=[keys_prefix]
            pre_keys = [key for p in keys_prefix for key in d.keys() if key.startswith(p)]
            do_keys = [key for key in pre_keys if key not in do_keys] + do_keys
        do_keys = [key for key in do_keys if not key in hide_keys]
        hkp = hide_keys_prefix;
        hkp = [hkp] if (type(hkp)==str) else hkp
        hkp = [key for p in hkp for key in do_keys if key.startswith(p)]
        do_keys = [key for key in do_keys if not key in hkp]
        return do_keys
    
    if y is None:
        d = x
        keys = set_keys(d, keys, keys_prefix, hide_keys, hide_keys_prefix)
        xvals = np.arange(len(yfunc(x[keys[0]]))) #may be inefficient for expensive yfunc.
        xvals = {key: (xvals) for key in keys}
    else:
        d = y
        keys = set_keys(d, keys, keys_prefix, hide_keys, hide_keys_prefix)
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
    failed_to_plot_keys = []
    for key in keys:
        try:
            plotter(xfunc(xvals[key]), yfunc(d[key]),
                    label=prefix+key+suffix, **kwargs)
        except:
            failed_to_plot_keys += [key]
    if failed_to_plot_keys != []:
        print("Warning: failed to plot for keys: "+', '.join(failed_to_plot_keys))
    plt.legend()



#### imshow functionality ####

def colorbar(im, ax=None, loc="right", size="5%", pad=0.05, label=None, **kwargs):
    """draws vertical colorbar with decent size and positioning to the right of data.
    
    use via e.g. {im = plt.imshow(); colorbar(im)}.
    
    Parameters
    ----------
    loc : string
        location of colorbar. e.g. "right", "bottom".
        If "bottom", may want to also input orientation="horizontal".
    size : string
        width compared to image. e.g. "5%" means cbar_xaxis is 5% of img_xaxis if loc="right".
    pad : float
        padding (in inches?) between plot and colorbar.
    label : None, or string
        if passed, will use defaults for pqol.clabel() to label colorbar.
    **kwargs go to plt.colorbar()
    """
    ax = ax if ax is not None else plt.gca()
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    cbar = plt.colorbar(im, cax=cax, **kwargs)
    if label is not None: clabel(label)
    return cbar

def clabel(label, ax=None, rotation= -90, va='baseline', **kwargs):
    """labels active Axes as if it was a vertical colorbar to the right of data."""
    ax = ax if ax is not None else plt.gca()
    return ax.set_ylabel(label, rotation=rotation, va=va, **kwargs)

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



#### field of view ####

## imshow field of view ##
    
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

def data_overlap(ax=None, gridsize=(3,3)):
    """Determines number of data points overlapping each box in a grid.
    
    Use ax, or plt.gca() if ax is not provided.
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
    im = plt.imshow(overlap, extent=[*plt.gca().get_xlim(), *plt.gca().get_ylim()]);
    #   ^^extent parameter is necessary to make imshow align with other plots
    pqol.colorbar(im)
    """ 
    (xdata, ydata) = get_data(ax, combine=True)
    xaxdata = _xcoords_data_to_ax(xdata, ax)
    yaxdata = _ycoords_data_to_ax(ydata, ax)
    
    ym = (  np.arange(gridsize[0]))/gridsize[0]
    yx = (1+np.arange(gridsize[0]))/gridsize[0]
    xm = (  np.arange(gridsize[1]))/gridsize[1]
    xx = (1+np.arange(gridsize[1]))/gridsize[1]
    in_ybox = [[]]*gridsize[0]
    in_xbox = [[]]*gridsize[1]
    for i in range(gridsize[0]):
        in_ybox[i] = (ym[i] < yaxdata) & (yaxdata < yx[i])
    for j in range(gridsize[1]):
        in_xbox[j] = (xm[j] < xaxdata) & (xaxdata < xx[j])
    
    r  = np.zeros(gridsize)
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            r[i][j] = np.sum(in_xbox[j] & in_ybox[i])
    
    r = np.flip(r, axis=0) #since ymax (not ymin) comes first (at top of img.)
    return r


#### annotation ####

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

def hline(y, text=None, textparams=dict(), textloc=0.5, textanchor="start",
          textside="top", textmargin=TEXTBOX_MARGIN,
          xmin=0, xmax=1, yspec="data", xspec="axes", textspec="axes",
          ax=None, **kwargs):
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
    additional **kwargs get passed to ax.axhline
        
    Examples
    --------
    #Try this
    plt.plot(range(10))
    pqol.hline(3, "hello, world!", textloc=7, textanchor="start", textspec="data")
    pqol.vline(7)
    """
    ax = ax if ax is not None else plt.gca()
    y_d    = y if yspec=="data" else \
                (_ycoords_ax_to_data(y, ax=ax) \
                     if yspec=="axes" else None) #y in data coords
    (xmin_a, xmax_a) = (xmin, xmax) if xspec=="axes" else \
                (_xcoords_data_to_ax((xmin, xmax), ax=ax) \
                     if xspec=="data" else (None, None)) #xmin & xmax in ax coords
    
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
          ax=None, **kwargs):
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
        
def _xcoords_ax_to_data(x_axcoords, ax=None, xlim=None):
    """Converts x-axis ax coords to x-axis data coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(x_axcoords, axis="x", convertto="data", ax=ax, lim=xlim)

def _ycoords_ax_to_data(y_axcoords, ax=None, ylim=None):
    """Converts y-axis ax coords to y-axis data coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(y_axcoords, axis="y", convertto="data", ax=ax, lim=ylim)

def _xcoords_data_to_ax(x_datacoords, ax=None, xlim=None):
    """Converts x-axis data coords to x-axis ax coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(x_datacoords, axis="x", convertto="ax", ax=ax, lim=xlim)

def _ycoords_data_to_ax(y_datacoords, ax=None, ylim=None):
    """Converts y-axis data coords to y-axis ax coords, via _xy_ax_data_convert."""
    return _xy_ax_data_convert(y_datacoords, axis="y", convertto="ax", ax=ax, lim=ylim)

def _margin(margin, axis="x", ax=None):
    """converts margin as fraction of full data range into data value.
    
    returns magin * (Zmax - Zmin), where 'Z' == axis == 'x' or 'y'.
    
    ax is set to active plot if None is passed.
    """
    ax = ax if ax is not None else plt.gca()
    lim = ax.get_xlim() if axis=="x" else (ax.get_ylim() if axis=="y" else None)
    return margin * (lim[1] - lim[0])


#### Save Plots ####
    
## Current Directory ##
    
savedir = os.getcwd()+'/saved_plots/' if 'savedir' not in locals().keys() else locals()['savedir']
"""default location for plot saves. Change using set_savedir(new_savedir)"""

## Save plots ##

def set_savedir(new_savedir):
    """sets savedir to the new_savedir.
    
    savedir is the default location for saved plots.
    it defaults to os.getcwd()+'/saved_plots/'.
    access via: import PlotQOL as pqol; pqol.savedir
    """
    global savedir
    savedir = new_savedir

def _convert_to_next_filename(name, folder=savedir):
    """returns string for next filename starting with name, in folder.
    
    Folder will be part of return; i.e. result contains the full path.
    If name starts with '/', ignores the 'folder' kwarg.
    Attempts to make folder if implied folder does not exist.
    Convention is to use filenames in order: [name, name 1, name 2, ...].
    Will not re-use earlier names if files are deleted.
    e.g. if only [name, name 3, name 4] exist, will return name 5, not name 1.
    """
    split = " " #splits name and number.
    
    if name.startswith('/'):
        folder = name[ : (name.rfind('/')) ] +'/'
        result = name
    else:
        result = folder + name
    if not os.path.isdir(folder):
        os.mkdir(folder)

    l = [N.replace(name, '').replace(split, '').split('.')[0] \
            for N in os.listdir(folder) if N.startswith(name)]
    if not '' in l:
        return result #since default name does not already exist as file.
    else:
        l.remove('')
        x = np.max([int(N) for N in l]) if len(l)>0 else 0
        return result + split + str(x + 1)

def savefig(fname=None, folder=savedir, Verbose=True, **kwargs):
    """Saves figure via plt.savefig()
    
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
    Verbose : bool. Default: True
        If True, prints location to which the plot is saved.
    **kwargs are passed to plt.savefig
    """
    fname = fname if fname is not None else DEFAULT_SAVE_STR
    saveto = _convert_to_next_filename(fname, folder=folder)
    bbox_inches = kwargs.pop("bbox_inches", 'tight')
    plt.savefig(saveto, bbox_inches=bbox_inches, **kwargs)
    if Verbose: print("Active plot saved to",saveto)

#### References to matplotlib built-in colors, markers, etc. ####
#copied from matplotlib docs.

## Colors ##

def colormaps():
    """Displays all available matplotlib colormaps."""
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
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
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