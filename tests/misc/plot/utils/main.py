##
##  utils.py
##
##  A collection of utilities used in several of Kai Fox's scripts.
##

import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
from utils import legendlib
from operator import sub
import more_itertools
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import contextlib
import itertools
import functools
import argparse
import struct
import types
import copy
import time
import sys
import os
import io

__all__ = []




# ========================================================================================
# ---------------------------------------------------------------  Miscellaneous tools  --
# ========================================================================================



# --------------------------------------------------------  Truly Misc  ----
# ==========================================================================


# Unified line_profiling
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
def enable_profiler():
    atexit.register(profile.print_stats)
__all__.append("enable_profiler")
__all__.append("profile")



class nullcontext(contextlib.AbstractContextManager):
    """Context manager that does no additional processing.
    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:
    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass
__all__.append('nullcontext')



# (Directly from argparse source)
class Namespace(argparse._AttributeHolder):
    """Simple object for storing attributes.
    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    # Added from argparse source
    def as_dict(self):
        return copy.copy(self.__dict__)

    # Also added
    def merged_with(self, *namespaces):
        d = self.as_dict()
        for n in namespaces:
            d.update(n.__dict__)
        return Namespace(**d)

    # This was added too
    @staticmethod
    def merged(*namespaces):
        d = {}
        for n in namespaces:
            d.update(n.__dict__)
        return Namespace(**d)

__all__.append('Namespace')
argparse.Namespace = Namespace



def parser_from_dictionary(d, parser = None):
    '''Construct an ArgumentParser from a dictionary specification.

    #### Parameters:

    - `d` --- The specifying dictionary. Keys give argument names
        and should point to their own dictionary describing keyword
        arguments to ArgumentParser.add_argument.
    - `parser` --- A parser to add arguments to. If `None` a new
        ArgumentParser will be constructed and returned.'''

    if parser is None:
        parser = argparse.ArgumentParser()
    for arg_name, arg_kwargs in d.items():
        parser.add_argument(arg_name, **arg_kwargs)
    return parser

__all__.append('parser_from_dictionary')
argparse.from_dict = parser_from_dictionary



def parse_nan(x, dtype = np.float):
    '''Cast between arbitary dtypes, with failures resulting in np.nan'''
    try:
        return dtype(x)
    except ValueError:
        return np.nan
__all__.append("parse_nan")



# See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# -> online algorithm
class IncrementalStats():
    def __init__(self, shape = []):
        self._mean = np.zeros(shape)
        self._count = 0
        self._M2 = np.zeros(shape)

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(newValue):
        self._count += 1 
        delta = newValue - self._mean
        self._mean = self._mean + delta / self._count
        delta2 = newValue - self._mean
        self._M2 = self._M2 + delta * delta2

    # retrieve the mean, variance and sample variance from an aggregate
    def render(self):
        return self._mean, self._M2/self._count

__all__.append("IncrementalStats")



def rename_code_object(code_object, new_name):
    return types.CodeType(
            code_object.co_argcount, code_object.co_kwonlyargcount,
            code_object.co_nlocals, code_object.co_stacksize,
            code_object.co_flags, code_object.co_code,
            code_object.co_consts, code_object.co_names,
            code_object.co_varnames, code_object.co_filename, new_name,
            code_object.co_firstlineno, code_object.co_lnotab,
            code_object.co_freevars, code_object.co_cellvars)


def copy_func(f, name = None):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    new_code = rename_code_object(f.__code__, name or f.__code__.co_name)
    g = types.FunctionType(new_code,
                           copy.copy(f.__globals__),
                           name = name or copy.copy(f.__name__),
                           argdefs = copy.copy(f.__defaults__),
                           closure = copy.copy(f.__closure__))
    g = functools.update_wrapper(g, f)
    g.__name__ = name or copy.copy(f.__name__)
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g



def trim_doc(docstring):
    '''Standard docstring formatting as defined in PEP 257.'''
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    maxint = 10 ** 10
    indent = maxint
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < maxint:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)
__all__.append("trim_doc")
dedent = trim_doc
__all__.append("dedent")


def isstringlike(obj):
    return isinstance(obj, str) or isinstance(obj, bytes)
__all__.append("isstringlike")

# --------------------------------------------------  Iterable Helpers  ----
# ==========================================================================


def isiterable(obj):
    '''Check if an object is iterable.'''
    return isinstance(obj, collections.Iterable)
__all__.append('isiterable')



def product(*lists, join = None, join_order = None):
    return tuple(xproduct(*lists, join = join, join_order = join_order))
__all__.append("product")

def xproduct(*lists, join = None, join_order = None):
    '''Variant of itertools.product allowing joining of combinations.
    Examples:

      > utils.product([1,2,3], ['a', 'b'], join='_')
      ['1_a', '1_b', '2_a', '2_b', '3_a', '3_b']

      > utils.product([1,2,3], ['a', 'b'], join='_', join_order = (1, 0))
      ['a_1', 'b_1', 'a_2', 'b_2', 'a_3', 'b_3']
    '''
    if join_order is None:
        join_order = range(len(lists))

    prod = itertools.product(*lists)
    if join is None:
        return list(prod)
    else:
        for combination in prod:
            yield join.join([str(combination[i]) for i in join_order])
__all__.append("xproduct")


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
__all__.append("flatten")


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def split(seq, sep, exclude = False):
    return tuple(xsplit(seq, sep, exclude))
__all__.append("split")

def xsplit(seq, sep, exclude = False):
    g = ()
    for el in seq:
        if el == sep:
            yield g
            g = ()
            if not exclude: yield (el,) 
        else:
            g = g + (el,)
    yield g
__all__.append("split")


# ----------------------------------------------------  Dict Utilities  ----
# ==========================================================================


def merge_dicts(*dicts):
    '''Merge dictionaries. In case of overlap, the dictionary that comes first 
    in the arguments is given precidence.'''
    dicts = list(dicts[::-1])
    main_dict = copy.copy(dicts.pop(0))
    for d in dicts:
        main_dict.update(d)
    return main_dict
__all__.append("merge_dicts")


def pop(x, key, default):
    try:
        return x.pop(key)
    except (IndexError, KeyError):
        return default
__all__.append('pop')


class ItemwiseDefaultDict(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.__defaults = {}

    def setdefault(self, key, value):
        self.__defaults[key] = value

    def popdefault(self, key):
        if key in self.__defaults:
            return self.__defaults.pop(key)
        else:
            raise KeyError("Key {} does not have a default.".format(repr(key)))

    def getdefault(self, key):
        if key in self.__defaults:
            return self.__defaults[key]
        else:
            raise KeyError("Key {} does not have a default.".format(repr(key)))

    def updatedefaults(self, new_defaults):
        self.__defaults.update(new_defaults)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        elif key in self.__defaults:
            return self.__defaults[key]
        else:
            raise KeyError(("Key {} is not present and "
                            "does not have a default.").format(repr(key)))
__all__.append("ItemwiseDefaultDict")





# ========================================================================================
# ---------------------------------------------------------------  Plotting Mechanisms  --
# ========================================================================================


# --------------------------------------------------  General Tools  ----
# =======================================================================


def force_subplots_adjust(fig, left = 0., right = 1.,
                          bottom = 0., top = 1.,
                          width = 1., height = 1.):
    '''Subplot adjustment when axes are not arranged in a grid structure.

    #### Parameters

    - `fig` --- The figure whose subplots should be adjusted.
    - `left`, `right`, `bottom`, `top` --- Positions in Figure coordinates
      for the new bounds of the figure.
    - `width`, `height` --- Multiplicative scaling factors.
    '''

    left_factor = left
    bottom_factor = bottom
    width_factor = width * (right - left)
    height_factor = height * (top - bottom)
    for ax_to_adjust in fig.get_axes():
        curr = ax_to_adjust.get_position().bounds
        ax_to_adjust.set_position([curr[0] * width_factor + left_factor,
                                   curr[1] * height_factor + bottom_factor,
                                   curr[2] * width_factor,
                                   curr[3] * height_factor])
__all__.append('force_subplots_adjust')


    
def give_axes_right_space(axes, space, pad = 0, adjust_width = True):
    '''Allocate space at the right of a plot for an axis or set of axes.

    #### Parameters

    - `axes` --- The matplotlib Axes instance being put at the right of
      the plot. Multiple Axes objects can be passed in an iterable; their
      height and vertical position will be preserved, but they will all
      be adjusted to fill the new space (if adjust_width is True).
    - `space` --- Width (in figure coords) of the space at the right of
      the plot to be allocated for the axes.
    - `pad` --- Padding (in figure coords) to be placed between the axes
      being moved to create space and the axes being placed at right.
    - `adjust_width` --- If true, the width of each axis in `axes` will
      be adjusted to fill the new space.'''

    if not isiterable(axes):
        axes = [axes]
    fig = axes[0].get_figure()
    fig.canvas.draw_idle()

    factor = 1 - space - pad
    force_subplots_adjust(fig, width = factor)
    
    # Put ax on the right with correct space
    for ax in axes:
        curr = ax.get_position().bounds
        width = space if adjust_width else curr[2]
        ax.set_position([factor + pad, curr[1], width, curr[3]])
__all__.append('give_axes_right_space')



def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio
__all__.append("get_aspect")



# Stolen from: https://stackoverflow.com/q/19394505
def points_from_data_units(distance, axis, reference='y', inverse = False):
    '''
    Convert a distance in data units to the same in points.

    Parameters
    ----------
    distance: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    points: float
        Linewidth in points
    '''
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    # For some reason this is really supposed to be 72, not fig.dpi
    length *= 72
    # Scale distance to value range
    return distance * (length / value_range) ** (-1 if inverse else 1)
__all__.append("points_from_data_units")

# --------------------------------------  PyPlot Style Abstractions  ----
# ======================================================================= 

class Style:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    def plot(self, x, y, lkw = {}, mkw = {}):
        if self.l: plt.plot(x, y, self.ls, color = self.lc, lw = 1, **lkw)
        if self.m: plt.plot(x, y, self.ms, color = self.mc, ms = 4, **mkw)

Style.teal     = Style(l = True,  lc="#bbccaa", ls='--',
                       m = True,  mc="#5599aa", ms='o')
Style.teal2    = Style(l = True,  lc="#5599aa", ls='-',
                       m = False)
Style.orange   = Style(l = True,  lc="#ccccbb", ls='--',
                       m = True,  mc="#dd9900", ms='o')
Style.orange2  = Style(l = True,  lc="#dd9900", ls='-',
                       m = False)
Style.violet   = Style(l = True,  lc="#ccbbcc", ls='--',
                       m = True,  mc="#aa0044", ms='o')
Style.violet2  = Style(l = True,  lc="#aa0044", ls='-',
                       m = False)

__all__.append("Style")

        
def plot_touch(touch, style):
        if isinstance(touch, pd.DataFrame):
                if style.l: plt.plot(touch['y'], touch['x'], style.ls, color = style.lc, lw = 1)
                if style.m: plt.plot(touch['y'], touch['x'], style.ms, color = style.mc, ms = 4)
        else:
                touch = np.array(touch)
                if style.l: plt.plot(touch[:,1], touch[:,0], style.ls, color = style.lc, lw = 1)
                if style.m: plt.plot(touch[:,1], touch[:,0], style.ms, color = style.mc, ms = 4)
__all__.append("plot_touch")




def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
            cmap : The matplotlib colormap to be altered
            start : Offset from lowest point in the colormap's range.
                    Defaults to 0.0 (no lower ofset). Should be between
                    0.0 and `midpoint`.
            midpoint : The new center of the colormap. Defaults to 
                    0.5 (no shift). Should be between 0.0 and 1.0. In
                    general, this should be  1 - vmax/(vmax + abs(vmin))
                    For example if your data range from -15.0 to +5.0 and
                    you want the center of the colormap at 0.0, `midpoint`
                    should be set to  1 - 5/(5 + 15)) or 0.75
            stop : Offset from highets point in the colormap's range.
                    Defaults to 1.0 (no upper ofset). Should be between
                    `midpoint` and 1.0.
        '''
        cdict = {
                'red': [],
                'green': [],
                'blue': [],
                'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
                np.linspace(0.0, midpoint, 128, endpoint=False), 
                np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
                r, g, b, a = cmap(ri)

                cdict['red'].append((si, r, r))
                cdict['green'].append((si, g, g))
                cdict['blue'].append((si, b, b))
                cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap
__all__.append('shiftedColorMap')



__xsmallfont = FontProperties()
__xsmallfont.set_size(10)


legend_styles = {
    'ggplot': {
        'loc': 'center left', 'bbox_to_anchor': (1, 0.5),
        'frameon': False},
    'minimal': {
        'frameon': False,
        'prop': __xsmallfont,
        'handlelength': 1},
}
__all__.append('legend_styles')



binary_palette = ["#3F51B5", "#9a1818"]
__all__.append('binary_palette')



def get_color(artist):
    '''Get the primary color of an arbitrary artist.'''
    if isinstance(artist, matplotlib.collections.PathCollection):
        return artist.get_facecolor()[0 ]
    elif isinstance(artist, matplotlib.lines.Line2D):
        return artist.get_color()
    else:
        raise ValueError("Type {} is not supported by get_color.".format(type(artist)))






# ---------------------------------------------------  Seaborn Mods  ----
# ======================================================================= 



sns.divergingN_palette = lambda n, *a, **kw: sns.diverging_palette(*a, **kw, n = n)


from seaborn.relational import _LinePlotter

class _LinePlotter_Override(_LinePlotter):

    # Redefine the data aggregation function for custom CI
    def aggregate(self, vals, grouper, units=None):
        # Parent class will get confused if self.ci is a function, so temporarily disable
        old_ci = self.ci
        if callable(self.ci):
            self.ci = None

        # Calculate as the parent class would
        x, y, y_ci = _LinePlotter.aggregate(self, vals, grouper, units = units)

        # If a callable was provided as the CI method, use that
        self.ci = old_ci
        if callable(self.ci):
            grouped = vals.groupby(grouper, sort=self.sort)
            cis = grouped.agg(self.ci)
            y_ci = pd.DataFrame(np.c_[y - cis, y + cis],
                                index = x,
                                columns=["low", "high"])

        return x, y, y_ci




    # Allow customization of the legend
    def add_legend_data(self, ax):
        """Add labeled artists to represent the different plot semantics.

        Legend is now specified in an iterable containing at least one of
        'hue', 'size', 'style' and optionally containig the flag 'brief'
        """


        # -----------------------  Input checking  ----

        if self.legend in ["brief", "full"]:
            return _LinePlotter.add_legend_data(self, ax)

        try:
            # These are all of the cases that this override is build to handle
            if all(leg_type not in self.legend for leg_type in ['hue' , 'size', 'style']):
                raise ValueError
        except (TypeError, ValueError):
            err = ("`legend` must be either 'brief', 'full', or False " + 
                      "or an iterable containing at least one of " + 
                      "['hue', 'size', 'style'].")
            raise ValueError(err)


        # ----------------------  Helpers / Setup  ----

        legend_kwargs = {}
        keys = []

        title_kws = dict(color="w", s=0, linewidth=0, marker="", dashes="")

        n_leg_types = len([typ for typ in self.legend if typ in ['hue' , 'size', 'style']])

        def update(var_name, val_name, **kws):

            key = var_name, val_name
            if key in legend_kwargs:
                legend_kwargs[key].update(**kws)
            else:
                keys.append(key)

                legend_kwargs[key] = dict(**kws)


        # -------- Add a legend for hue semantics  ----

        if 'hue' in self.legend:
            if "brief" in self.legend and self.hue_type == "numeric":
                if isinstance(self.hue_norm, mpl.colors.LogNorm):
                    ticker = mpl.ticker.LogLocator(numticks=3)
                else:
                    ticker = mpl.ticker.MaxNLocator(nbins=3)
                hue_levels = (ticker.tick_values(*self.hue_limits)
                                    .astype(self.plot_data["hue"].dtype))
            else:
                hue_levels = self.hue_levels

            # Add the hue semantic subtitle
            if self.hue_label is not None and n_leg_types > 1:
                update((self.hue_label, "title"), self.hue_label, **title_kws)

            # Add the hue semantic labels
            for level in hue_levels:
                if level is not None:
                    color = self.color_lookup(level)
                    update(self.hue_label, level, color=color)


        # ------- Add a legend for size semantics  ----

        if 'size' in self.legend:
            if "brief" in self.legend  and self.size_type == "numeric":
                if isinstance(self.size_norm, mpl.colors.LogNorm):
                    ticker = mpl.ticker.LogLocator(numticks=3)
                else:
                    ticker = mpl.ticker.MaxNLocator(nbins=3)
                size_levels = (ticker.tick_values(*self.size_limits)
                                     .astype(self.plot_data["size"].dtype))
            else:
                size_levels = self.size_levels

            # Add the size semantic subtitle
            if self.size_label is not None and n_leg_types > 1:
                update((self.size_label, "title"), self.size_label, **title_kws)

            # Add the size semantic labels
            for level in size_levels:
                if level is not None:
                    size = self.size_lookup(level)
                    update(self.size_label, level, linewidth=size, s=size)


        # ------ Add a legend for style semantics  ----

        if 'style' in self.legend:
            # Add the style semantic title
            if self.style_label is not None and n_leg_types > 1:
                update((self.style_label, "title"), self.style_label, **title_kws)

            # Add the style semantic labels
            for level in self.style_levels:
                if level is not None:
                    update(self.style_label, level,
                           marker=self.markers.get(level, ""),
                           dashes=self.dashes.get(level, ""))


        # -----------------  Organize legend data  ----

        func = getattr(ax, self._legend_func)

        legend_data = {}
        legend_order = []

        for key in keys:

            _, label = key
            kws = legend_kwargs[key]
            kws.setdefault("color", ".2")
            use_kws = {}
            for attr in self._legend_attributes + ["visible"]:
                if attr in kws:
                    use_kws[attr] = kws[attr]
            artist = func([], [], label=label, **use_kws)
            if self._legend_func == "plot":
                artist = artist[0]
            legend_data[label] = artist
            legend_order.append(label)

        self.legend_data = legend_data
        self.legend_order = legend_order


    def legendlib_data(self):
        if not hasattr(self, "legend_data"):
            self.add_legend_data(plt.gca())
        return {k: v.get_color() for k,v, in self.legend_data.items()}


# Also allow access to legendlib-compatible data for when sns.lineplot
# is called instead of the utils version
setattr(sns.relational._LinePlotter, "legendlib_data",
        _LinePlotter_Override.legendlib_data)



def lineplot(x=None, y=None, hue=None, size=None, style=None, data=None,
             palette=None, hue_order=None, hue_norm=None,
             sizes=None, size_order=None, size_norm=None,
             dashes=True, markers=None, style_order=None,
             units=None, estimator="mean", ci=95, n_boot=1000,
             sort=True, err_style="band", err_kws=None,
             legend="brief", legend_kws = None, force_legend_data = False,
             ax=None, title = None, legend_title = None, **kwargs):

    '''Duplicate of seaborn.lineplot, but allows use of callable for determining error.'''

    p = _LinePlotter_Override(
        x=x, y=y, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        dashes=dashes, markers=markers, style_order=style_order,
        units=units, estimator=estimator, ci=ci, n_boot=n_boot,
        sort=sort, err_style=err_style, err_kws=err_kws, legend=False
    )

    if ax is None:
        ax = plt.gca()
    p.plot(ax, kwargs)

    if title is not None:
        ax.set_title(title)

    # Enable custom legends
    if legend:
        p.legend = legend
        p.add_legend_data(ax)

        legend_kws = legend_kws or {}
        legend_kws.setdefault("title", legend_title)

        if 'llcovariate' in p.legend:
            ax.set_position([0, 0, 1, 1])
            legendlib.covariate_legend(ax.get_figure(), p.legendlib_data(), **legend_kws)
        elif 'llstandard' in p.legend:
            ax.set_position([0, 0, 1, 1])
            legendlib.standard_legend(ax.get_figure(), p.legendlib_data(), **legend_kws)
        else:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(**legend_kws)

    return ax, p
__all__.append('lineplot')







# Redefines the seaborn functions stripplot, swarmplot, and boxplot,
# with legendlib integration.

plotter_save = {}
categorical_f_names = ["stripplot", "swarmplot", "boxplot"]
categorical_orig_fs = [sns.stripplot, sns.swarmplot, sns.boxplot]
categorical_classes = [sns.categorical._StripPlotter,
                       sns.categorical._SwarmPlotter,
                       sns.categorical._BoxPlotter]

def categorical_legendlib_data(self):
    if hasattr(self, "_legendlib_data"):
        return self._legendlib_data
    else:
        raise ValueError("Cannot get legendlib-compatible data " + 
                         "for this plotter object.")

# Create an override of the add_legend_data function for some of the
# seaborn categorical plotters. The conditional strucutre here is due
# to the inconsistency in what the add_legend_data function actually does
# for different plotters.
def categorical_add_legend_data(self, *args):
    if len(args) == 1:
        # The function was called as: plotter.add_legend_data(ax)
        # All plotters that don't normally accept this signature
        # do the same thing with their own add_legend_data function:
        for c, n in zip(self.colors, self.hue_names):
            sns.categorical._CategoricalPlotter.add_legend_data(args[0], c, n)
    else:
        # Called as add_legend_data(axes, color, name)
        # This is the nonstandard signature for seaborn. It *is* the
        # default for categorical plotters though
        sns.categorical._CategoricalPlotter.add_legend_data(*args)


for f_name, orig_f, klass in zip(categorical_f_names,
                                 categorical_orig_fs,
                                 categorical_classes):


    def categorical_override(*args, **kwargs):
        
        # Define a function that we can monkeypatch onto the plotter class
        # whenever legendlib legends are being used
        # These monkeypatching functions have to be defined in here so that
        # the classes are treated correctly.
        # This hacked function will save the plotter object for later access
        # and parse out legend data in a format friendly to legendlib
        old_legdata = klass.add_legend_data
        def add_legend_data_override(self, ax, *args, **kwargs):
            """Add empty scatterplot artists with labels for the legend."""
            # When this override is being called, it means we
            # want to save any relevant data for legendlib
            self._legendlib_data = {}
            for rgb, label in zip(self.colors, self.hue_names):
                self._legendlib_data[label] = matplotlib.colors.rgb2hex(rgb)
            self._legendlib_data = copy.copy(plotter_save[ax])
            plotter_save[ax] = self
            return old_legdata(self, ax, *args, **kwargs)

        # Another monkeypatching function...
        # Hides hue names so the plotter object won't try to make its own legend.
        old_annot = klass.annotate_axes
        def annotate_axes_override(self, ax):
            old_hue_names = self.hue_names
            self.hue_names = None
            result = old_annot(self, ax)
            self.hue_names = old_hue_names
            return result

        # Process the custom arguments
        # We don't want to confuse the original plotting function with them
        # so pop them from kwargs
        legend = pop(kwargs, "legend", None)
        legend_title = pop(kwargs, "legend_title", None)
        legend_kws = pop(kwargs, "legend_kws", None)

        # Do the normal seaborn behavior except with our legend overrides patched in
        if isinstance(legend, str) and (legend.startswith('ll') or legend == 'data'):    
            klass.add_legend_data = add_legend_data_override
            klass.annotate_axes = annotate_axes_override

        ax = orig_f(*args, **kwargs)
        plotter = plotter_save[ax].pop("__plotter__")
        if ax in plotter_save: plotter_save.pop(ax)

        if isinstance(legend, str) and legend.startswith('ll' or legend == 'data'):
            # Undo the patching so we don't break stuff permamently
            klass.add_legend_data = old_legdata
            klass.annotate_axes = old_annot

            # Apply the legend override
            ax.set_position([0, 0, 1, 1])
            legend_kws = legend_kws or {}
            legend_kws.setdefault("title", legend_title)
            if legend == "llstandard":
                legendlib.standard_legend(ax.get_figure(),
                                          plotter.legendlib_data(),
                                          **legend_kws)
            elif legend == "llcovariate":
                legendlib.covariate_legend(ax.get_figure(),
                                           plotter.legendlib_data(),
                                           **legend_kws)

        return ax, plotter

    exec(f_name + " = copy_func(categorical_override, name = '" + f_name + "')")
    __all__.append(f_name)

    # Allow access to the computed legendlib data
    setattr(klass, "legendlib_data", categorical_legendlib_data)
    if klass.add_legend_data is sns.categorical._CategoricalPlotter.add_legend_data:
        klass.add_legend_data = categorical_add_legend_data








# Add legendlib compatibility and some other custom functionality to the
# seaborn FacetGrid class.

def facetgrid_add_legend(grid, legend_data=None, title=None, label_order=None,
                         style = None, ll_kws = {}, **kwargs):
    '''Add a legend to a facetgrid.'''

    legend_data = legend_data or grid.aggregated_legendlib_data()
    try:
        legendlib.apply_legend(style, plt.gcf(), legend_data, title = title, **ll_kws)
    except ValueError:
        # `style` was not a legendlib legend type
        grid.add_legend(legend_data, title, label_order, **kwargs)

__all__.append("facetgrid_add_legend")



# Allow a seamless inferface for the above FacetGrid overrides
# (also convenient internal access)
class FacetGrid(sns.axisgrid.FacetGrid):

    def __init__(self, *args, **kwargs):
        sns.axisgrid.FacetGrid.__init__(self, *args, **kwargs)
        self._legendlib_data = {}

    add_legend = facetgrid_add_legend

    @property
    def ncol(self): return self._ncol
    @property
    def nrow(self): return self._nrow


    def isborder(self, r, c):
        ret = []
        if r == 0: ret.append('top')
        if c == 0: ret.append('left')
        if r == self._nrow - 1: ret.append('bottom')
        if c == self._ncol - 1: ret.append('right')
        return tuple(ret)


    def facet_ax_data(self):
        '''Iterate over facet columns and rows.'''

        data = self.data

        # Construct masks for the row variable
        if self.row_names:
            row_masks = [data[self._row_var] == n for n in self.row_names]
        else:
            row_masks = [np.repeat(True, len(self.data))]

        # Construct masks for the column variable
        if self.col_names:
            col_masks = [data[self._col_var] == n for n in self.col_names]
        else:
            col_masks = [np.repeat(True, len(self.data))]
        
        # Here is the main generator loop
        for (i, row), (j, col) in itertools.product(enumerate(row_masks),
                                                    enumerate(col_masks)):
            yield (i, j), row & col & self._not_na



    def map_custom(self, func, *args, **kwargs):
        # Differences from FacetGrid.map:
        # - Axis labels are not handled. Those are left to func
        # - Arguments to func are: [grid, row, column, data_mask, *args, *kwargs]
        # - Legend data gathering is handled differently, not as extensibly.
        # - Return value is an array of returns from func, not the FacetGrid


        # If color was a keyword argument, grab it here
        ax_labels = pop(kwargs, "ax_labels", None)

        if hasattr(func, "__module__"):
            func_module = str(func.__module__)
        else:
            func_module = ""

        # Check for categorical plots without order information
        if func_module == "seaborn.categorical":
            if "order" not in kwargs:
                warning = ("Using the {} function without specifying "
                           "`order` is likely to produce an incorrect "
                           "plot.".format(func.__name__))
                warnings.warn(warning)


        # Iterate over the data subsets
        returns = [[None for __ in range(self.ncol)] for _ in range(self.nrow)]
        for (row_i, col_j), data_mask in self.facet_ax_data():

            # If this subset is null, move on
            if not sum(data_mask):
                continue

            # Get the current axis
            ax = self.facet_axis(row_i, col_j)

            # Insert the other hue aesthetics if appropriate
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # Draw the plot
            result = func(self, row_i, col_j, data_mask, *args, **kwargs)


            # Update the legendlib data to reflect this new axis
            if ( isiterable(result) and
                 len(result) > 1 and
                 hasattr(result[1], "legend_data") ):
                self._legend_data.update(result[1].legend_data)

            returns[row_i][col_j] = result

        # Finalize the annotations and layout
        if ax_labels is not None:
            self._finalize_grid(ax_labels)

        return returns


    def _update_legend_data_plotter(self, plotter):
        # Update the legend data to reflect this new axis
        if hasattr(plotter, "legend_data"):
            self._legend_data.update(plotter.legend_data)
        else:
            raise ValueError("Provided plotter passed has no legend_data function.")


    def aggregated_legend_data(self):
        return self._legend_data

    def aggregated_legendlib_data(self):
        return {k: get_color(v) for k,v in self._legend_data.items()}

__all__.append("FacetGrid")






# Update the seaborn HeatMap function to allow adding a legendlib colorbar 

def heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            linewidths=0, linecolor="white",
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=False, xticklabels="auto", yticklabels="auto",
            mask=None, ax=None, **kwargs):

    # Stop automatic plotting of colorbar if a legendlib colorbar was requested
    if isstringlike(cbar) and cbar.startswith("ll"):
        old_cbar = cbar
        cbar = False

    # Initialize the plotter object
    plotter = sns.matrix._HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws, xticklabels,
                          yticklabels, mask)

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    
    # Restore old colorbar setting
    if isstringlike(cbar) and cbar.startswith("ll"):
        plotter.cbar = old_cbar
        ll.vcolorbar_legend(plt.gcf(), plotter.legendlib_data(), **cbar_kws)

    return ax, plotter
__all__.append("heatmap")


def heatmap_legendlib_data(plotter):
    return (plotter.cmap, plotter.vmin, plotter.vmax)
sns.matrix._HeatMapper.legendlib_data = heatmap_legendlib_data






# ========================================================================================
# -------------------------------------------------------------  BeuatifulSoup Helpers  --
# ========================================================================================



def check_for_classes(*classes):
    '''A non-sensitive check for classes in an unparsed class string.
    The main use case is in SoupStrainers.'''

    def perform_check(class_string):
        if class_string is None: return False
        class_string = class_string.split()
        return all([cls in class_string for cls in classes])
    return perform_check
__all__.append("check_for_classes")





# ========================================================================================
# --------------------------------------------------------------------  Pandas Helpers  --
# ========================================================================================

def insert_rows(df, ix, value):
    '''Insert row(s) into a dataframe before index ix.
    This function only works with integer-indexed DataFrames.

    #### Parameters
    - `df` --- The dataframe to insert into.
    - `ix` --- The index at which to insert.
    - `value` --- A pandas DataFrame object containing values of
        the new row(s). To obtain such a dataframe from a Series,
        you can run series.to_frame().transpose()'''

    before = df.ix[:ix-1]
    after = df.ix[ix:]
    return pd.concat([before, value, after])

__all__.append("insert_rows")
pd.DataFrame.insert_rows = insert_rows



def empty_dataframe(index, columns, dtypes):
    df = pd.DataFrame(index = index)
    for c, t in zip(columns, dtypes):
        df[c] = np.empty(len(index), dtype = t)
    return df
__all__.append("empty_dataframe")

@staticmethod
def __empty_dataframe(index, columns, dtypes):
    return empty_dataframe(index, columns, dtypes)
pd.DataFrame.empty = __empty_dataframe


def unique_rows(df, *columns):
    '''Select rows of a dataframe that are unique in the given columns.
    If `None` is provided as a column or no columns are given, the
    comparison will be across all columns.'''

    if None in columns or len(columns) == 0:
        columns = df.columns
    unique = {}
    for i,r in df[list(columns)].iterrows():
        unique[tuple(r)] = i
    return df.iloc[list(unique.values())]
__all__.append("unique_rows")
pd.DataFrame.unique_rows = unique_rows


