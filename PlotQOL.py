#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:52:21 2020

@author: Sevans

Useful and user-friendly/convenient codes for making matplotlib plots.
"""

#TODO: implement title for colorbar()
#TODO: implement scatter plot marker cycle
    
#check out illustrator

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.stats import linregress



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
    
set_plot_defaults()

XYLIM_MARGIN=0.05



#### slice string interpreter ####
    #Not intrinsically related to "plots" though. belongs in different QOL file?
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
    #Try the following:
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
    data = data if data is not None else get_data(ax)
    xlim = xlim if xlim is not True else ax.get_xlim()
    ylim = ylim if ylim is not True else ax.get_ylim()
    x    = np.array([x for l in data for x in l[0]])
    y    = np.array([y for l in data for y in l[1]])
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

def get_data(ax=None):
    """gets data of anything plotted by plt.plot & plt.scatter, on ax.
    
    if ax is None, defaults to current axes (i.e. active plot).
    returns:
        list of [xdata, ydata] arrays (potentially masked).
        .plot arrays will be listed first, followed by .scatter masked arrays.
    (for static return formatting, not based on output, use _get_alldata.)
    """
    d = _get_alldata(ax)
    return d["plot"] + d["scatter"]
        
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



#### References to matplotlib colors ####
#copied from matplotlib docs.

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