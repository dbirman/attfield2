from matplotlib.font_manager import FontProperties
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
import matplotlib.textpath as mattp
import matplotlib.colorbar as matcb
import matplotlib.colors as matcol
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import utils




def covariate_legend(fig, legend_data, title = None, line_height = 2.5, top_margin = 2,
                     column_margin = 2, marker_length = 1., left_margin = 2,
                     font_size = 10, font_family = 'DejaVU Sans', marker_kws = {}):
    '''Add a legend that describes covariates in a table.

    So far, this method only supports color-based legends. In the future,
    size, marker type, and dash wouldn't be too hard to implement. All size
    parameters (except font_size) are expressed as multiples of the font's
    height.

    ##### Example Usage:

        fmri = sns.load_dataset("fmri")
        split_names = ("event", "region")
        fmri['__split'] = list(zip(*[fmri[col] for col in split_names]))

        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize = (7, 4))
            ax, p = utils.lineplot(x="timepoint", y="signal", hue="__split",
                                   data=fmri, err_style = 'band',
                                   legend = ['hue', 'llcovariate'], legend_title = split_names)
            ax.set_title("Example Plot")
            utils.force_subplots_adjust(fig, left = 0.12, bottom = 0.16, right = 0.96, top = 0.9)
            plt.show()


    #### Parameters

    - `fig` --- The figure to add the legend to.
    - `legend_data` --- A dictionary mapping tuples to colors. The tuples
        should contain values of the covariates in a standard order.
    - `title` --- An iterable containing names of the covariates in the
        same order as they are provided in the `legend_data` dictionary.
    - `line_height` --- Total line height of a legend entry (times text_height).
    - `column_margin` --- Space between labels and markers (times text_height).
    - `left_margin` --- Space between markers and the body of the plot (times
        text_height).
    - `top_margin` --- Vertical position for the top of the lengend (times
        text_height).
    - `marker_length` --- Length of the marker line (times text_height).
    - `font_size` --- Size of all text in points. Passing `None` uses the default
        from matplotlib's rc.
    - `font_family` --- Font family for all text. Passing `None` uses the default
        from matplotlib's rc.
    - `marker_kws` --- Keywords for the pyplot.plot call for drawing markers.'''


    font_size = font_size or rcParams['font.size']
    font_family = font_family or rcParams['font.family']

    prev_ax = plt.gca()    
    ax = plt.axes([1, 0, 1, 1])
    ax.axis('off')
    
    # Set data coordinates equivalent to inches
    figsize = ax.get_figure().get_size_inches()
    ax.set_xlim(-1, figsize[0] - 1)
    ax.set_ylim(-figsize[1], 0)
    
    
    disp2data = ax.transData.inverted().transform
    text_height = 0
    column_x = 0
    
    
    # Write the columns
    for c in range(len(list(legend_data.keys())[0])):
        
        text_objs = []
        curr_col_width = 0
    
        # Draw all the objects and calculate size information
        for key, artist in legend_data.items():
            text_objs += [plt.text(0, 0, key[c], ha = 'center', va = 'top',
                                   size = font_size, family = font_family)]
            fig.canvas.draw()

            box = disp2data(text_objs[-1].get_window_extent())
            curr_col_width = max(curr_col_width, abs(box[1, 0] - box[0, 0]))
            
            # If we haven't yet measured the text height, apply that to all the
            # parameters that are expressed in terms of text size
            if text_height == 0:
                text_height = abs(box[1, 1] - box[0, 1])
                top_margin *= text_height
                left_margin *= text_height
                column_margin *= text_height
                line_height *= text_height
                marker_length *= text_height
                

        # Move all the objects to their correct position using the size information from above
        for i, text_obj in enumerate(text_objs):
            text_obj.set_position([column_x + curr_col_width / 2, -top_margin - line_height*i])
            
        # Plot the name of the split
        if title is not None:
            label = plt.text(column_x + curr_col_width / 2,
                             -top_margin - line_height*len(text_objs),
                             title[c], ha = 'right', va = 'top',
                             rotation = 75, rotation_mode = "anchor",
                             size = font_size, family = font_family)
            label.set_weight('bold')
        
        # Update x for next column based on the width of this one
        column_x += curr_col_width + column_margin
        
        
    # Draw the color lines
    for i, color in enumerate(legend_data.values()):
        marker_kws['lw'] = marker_kws.get("lw", 1)
        ax.plot([-column_margin, -column_margin - marker_length],
                [-top_margin - line_height*i - text_height / 2] * 2,
                color = color)
    
    
    
    # Set minimal x limits
    border_points_data = [[column_x - column_margin, 0],
                          [- left_margin - column_margin - marker_length, 0]]

    # Define transformation that we'll use to figure out requested space
    data2display = ax.transData.transform
    display2fig = fig.transFigure.inverted().transform
    
    # Push the main plot in so there's space for the legend axes
    border_points_in = display2fig(data2display(border_points_data))
    requested_space = abs(border_points_in[0, 0] - border_points_in[1, 0])

    ax.set_xlim(border_points_data[1][0], border_points_data[0][0])
    utils.give_axes_right_space(ax, requested_space)

    plt.sca(prev_ax)







def standard_legend(fig, legend_data, title = None, line_height = 2.5, column_margin = 1.5,
                    left_margin = 2, top_margin = 2, marker_length = 1.,
                    font_size = 10, font_family = 'DejaVU Sans', marker_kws = {}):
    '''Add standard legend, but in the style of legendlib.

    So far, this method only supports color-based legends. In the future,
    size, marker type, and dash wouldn't be too hard to implement. All size
    parameters (except font_size) are expressed as multiples of the font's
    height.

    ##### Example Usage:

        fmri = sns.load_dataset("fmri")

        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize = (7, 4))
            ax, p = utils.lineplot(x="timepoint", y="signal", hue="region",
                                   data=fmri, err_style = 'band',
                                   legend = ['hue', 'llstandard'], legend_title = "region")
            ax.set_title("Example Plot")
            utils.force_subplots_adjust(fig, left = 0.13, bottom = 0.16, right = 0.96, top = 0.9)
            plt.show()


    #### Parameters

    - `fig` --- The figure to add the legend to.
    - `legend_data` --- A dictionary mapping labels to colors.
    - `title` --- A title to print in bold above the main part of the legend.
    - `line_height` --- Total line height of a legend entry (times text_height).
    - `column_margin` --- Space between labels and markers (times text_height).
    - `left_margin` --- Space between markers and the body of the plot (times
        text_height).
    - `top_margin` --- Vertical position for the top of the lengend (times
        text_height).
    - `marker_length` --- Length of the marker line (times text_height).
    - `font_size` --- Size of all text in points. Passing `None` uses the default
        from matplotlib's rc.
    - `font_family` --- Font family for all text. Passing `None` uses the default
        from matplotlib's rc.
    - `marker_kws` --- Keywords for the pyplot.plot call for drawing markers.'''


    font_size = font_size or rcParams['font.size']
    font_family = font_family or rcParams['font.family']
    
    prev_ax = plt.gca()
    ax = plt.axes([1, 0, 1, 1])
    ax.axis('off')

    # Set data coordinates equivalent to inches
    figsize = ax.get_figure().get_size_inches()
    aspect = ax.get_position().width / ax.get_position().height
    ax.set_ylim(-figsize[1], 0)
    ax.set_xlim(-1, figsize[0] - 1)


    disp2data = ax.transData.inverted().transform
    text_height = 0
    col_width = 0
    text_objs = []

    # Draw all the objects and calculate size information
    for i, (key, artist) in enumerate(legend_data.items()):
        text_objs += [ax.text(0, 0, key, ha = 'center', va = 'top', fontproperties = FontProperties(
                              size = font_size, family = font_family))]
        
        fig.canvas.draw()
        box = disp2data(text_objs[-1].get_window_extent(ax.get_renderer_cache()))
        col_width = max(col_width, abs(box[1, 0] - box[0, 0]))

        # If we haven't yet measured the text height, apply that to all the
        # parameters that are expressed in terms of text size
        if text_height == 0:
            text_height = abs(box[1, 1] - box[0, 1])
            top_margin *= text_height
            left_margin *= text_height
            column_margin *= text_height
            line_height *= text_height
            marker_length *= text_height

    # Draw the title and get width info
    if title is not None:
        text_obj = ax.text(-column_margin - marker_length, -top_margin, title,
                           ha = 'left', va = 'top', size = font_size,
                           family = font_family, weight = 'bold')

        fig.canvas.draw()
        box = disp2data(text_obj.get_window_extent(ax.get_renderer_cache()))
        total_width = max(col_width, box[1, 0])
    else:
        total_width = col_width

    # Move all the objects to their correct position using the size information from above
    for i, text_obj in enumerate(text_objs):
        if title is not None: i += 1
        text_obj.set_position([col_width / 2, -top_margin - line_height*i])

    # Draw the color lines
    for i, color in enumerate(legend_data.values()):
        if title is not None: i += 1
        marker_kws['lw'] = marker_kws.get("lw", 1)
        ax.plot([-column_margin, -column_margin - marker_length],
                [-top_margin - line_height*i - text_height / 2] * 2,
                color = color)

    # Set minimal x limits
    border_points_data = [[total_width, 0],
                          [- left_margin - column_margin - marker_length, 0]]

    # Define transformation that we'll use to figure out requested space
    data2display = ax.transData.transform
    display2fig = fig.transFigure.inverted().transform
    
    # Push the main plot in so there's space for the legend axes
    border_points_in = display2fig(data2display(border_points_data))
    requested_space = abs(border_points_in[0, 0] - border_points_in[1, 0])

    ax.set_xlim(border_points_data[1][0], border_points_data[0][0])
    utils.give_axes_right_space(ax, requested_space)

    plt.sca(prev_ax)









def vcolorbar_legend(fig, legend_data, title = None,
                     left_margin = 2, right_margin = 2,
                     top_margin = 2, bottom_margin = 2, column_margin = 2,
                     ttl_indent = -0.2, ttl_line_height = 2, bar_width = 1.3,
                     font_size = 10, font_family = 'DejaVU Sans',
                     colorbar_kws = {}):

    '''Add standard vertical colorbar, but in the style of legendlib.

    ##### Example Usage:

        normal_data = np.random.randn(10, 12)

        with sns.axes_style("ticks"):
            ax, p = utils.heatmap(normal_data, center=0, cbar = 'll')
            ax.set_position([0, 0, 1, 1])
            ll.vcolorbar_legend(plt.gcf(), p.legendlib_data(), title = "Intensity")
            utils.force_subplots_adjust(plt.gcf(),
                    left = 0.11, right = 0.95,
                    bottom = 0.13, top = 0.89)
            plt.show()


    #### Parameters

    - `fig` --- The figure to add the legend to.
    - `legend_data` --- A tuple of the form (colormap, vmin, vmax).
    - `title` --- A title to print in bold above the main part of the legend.
    - `line_height` --- Total line height of a legend entry (units of text_height).
    - `left_margin` --- Space between colorbar and the body of the plot (units of
        text_height).
    - `right_margin` --- Space between colorbar and right edge of the plot (units of
        text_height).
    - `top_margin` --- Vertical position for the top of the lengend, measured from
        the top of all existing axes. (units of text_height).
    - `bottom_margin` --- Vertical position for the bottom of the lengend, measured
        from the bottom of all existing axes. (units of text_height).
    - `column_margin` --- Space between labels and markers (units of text_height).
    - `ttl_indent` --- Indentation of the title from the left side of the colorbar
        (units of text_height). Negative values indicate dedent.
    - `ttl_line_height` --- Line height of the title text (units of text_height).
    - `bar_width` --- Width of the color bar (units of text_height).
    - `font_size` --- Size of all text in points. Passing `None` uses the default
        from matplotlib's rc.
    - `font_family` --- Font family for all text. Passing `None` uses the default
        from matplotlib's rc.
    - `colorbar_kws` --- Keyword arguments to pyplot.ColorbarBase.__init__.'''

    vcolorbars_legend(fig, [legend_data], title = None if title is None else [title],
        left_margin = left_margin, right_margin = right_margin,
        top_margin = top_margin, bottom_margin = bottom_margin, column_margin =  column_margin,
        ttl_indent = ttl_indent, ttl_line_height = ttl_line_height, bar_width = bar_width,
        font_size = font_size, font_family = font_family,
        colorbar_kws = colorbar_kws)




def vcolorbars_legend(fig, legend_data, title = None,
                      left_margin = 2, right_margin = 2,
                      top_margin = 2, bottom_margin = 2, column_margin = 2,
                      ttl_indent = -0.2, ttl_line_height = 2, bar_width = 1.5,
                      font_size = 10, font_family = 'DejaVU Sans',
                      colorbar_kws = {}):
    '''Add multiple vertical colorbars in the style of legendlib.

    ##### Example Usage:

        normal_data = np.random.randn(10, 12)

        with sns.axes_style("ticks"):
            ax, p = utils.heatmap(normal_data, center=0, cbar = 'll')
            ax.set_position([0, 0, 1, 1])
            ll.vcolorbars_legend(plt.gcf(),
                [p.legendlib_data(), (matplotlib.cm.YlGnBu, 0, 1)],
                title = ["Intensity", "Variance"])
            utils.force_subplots_adjust(plt.gcf(),
                left = 0.11, right = 0.95,
                bottom = 0.13, top = 0.89)
            plt.show()


    #### Parameters
    
    All parameters are the same as those for legendlib.vcolorbar_legend, except
    for the changes noted here.

    - `legend_data` --- A list of tuples of the form (colormap, vmin, vmax).
    - `title` --- A list of title to print in bold above each colorbar legend.'''


    prev_ax = plt.gca()

    parents = fig.axes
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    font_size = font_size or rcParams['font.size']
    font_family = font_family or rcParams['font.family']
    if title is None:
        ttl_line_height = 0

    text_height = None
    column_widths = []
    colorbars = []

    for bar_data, ttl in zip(legend_data, title):

        cax = fig.add_axes((0, 0, 1, 1), label = ttl)
        cax.patch.set_visible(False)
        fig.canvas.draw()
        renderer = cax.get_renderer_cache()

        disp2figX = lambda x: fig.transFigure.inverted().transform([x, 0])[0] 
        disp2figY = lambda y: fig.transFigure.inverted().transform([0, y])[1] 

        if text_height is None:
            # Gather information about text size
            font = FontProperties(size = font_size, family = font_family)
            t = cax.text(0, 0, '1', font_properties = font)
            text_height = disp2figY(t.get_window_extent(renderer).height)
            t.remove()
            del t

            # Update size parameters from text_height units to display coords
            left_margin *= text_height
            right_margin *= text_height
            top_margin *= text_height
            bottom_margin *= text_height
            column_margin *= text_height
            ttl_line_height *= text_height
            bar_width *= text_height
            ttl_indent *= text_height

        # Set colorbar size
        top_idx = 3
        bottom_idx = 1
        cax_top = parents_bbox.extents[top_idx] - top_margin - ttl_line_height
        cax_bottom = parents_bbox.extents[bottom_idx] + bottom_margin
        cax.set_position([0, cax_bottom,
                          bar_width, cax_top - cax_bottom])

        # Create the actual colorbar 
        norm = matcol.Normalize(vmin = bar_data[1], vmax = bar_data[2])
        bar = matcb.ColorbarBase(cax, cmap = bar_data[0], norm = norm, **colorbar_kws)
        bar.outline.set_linewidth(0)

        # Set tick properties and get total width of bar
        max_x = 0
        for label in cax.get_yticklabels():
            label.set_fontproperties(font)
            right = label.get_window_extent(renderer).extents[2]
            max_x = max(max_x, right)


        disp2dataY = lambda y: cax.transData.inverted().transform([0, 0])[1] \
                             - cax.transData.inverted().transform([0, y])[1]
        disp2dataX = lambda x: cax.transData.inverted().transform([x, 0])[0]
        fig2dispY = lambda y: fig.transFigure.transform([0, y])[1]
        fig2dispX = lambda x: fig.transFigure.transform([x, 0])[0]

        # Add title
        if ttl is not None:
            title_y = disp2dataY(fig2dispY(ttl_line_height))
            title_x = disp2dataX(fig2dispX(ttl_indent))
            t = cax.text(title_x, 1-title_y, ttl, ha = 'left', va = 'top',
                         font_properties = font)
            t.set_weight("bold")

            # Update max_x if the title is wider than the colorbar
            right = t.get_window_extent(renderer).extents[2]
            max_x = max(max_x, right)

        column_widths.append(disp2figX(max_x))
        colorbars.append(cax)


    total_width = sum(column_widths) + right_margin
    utils.give_axes_right_space(colorbars, total_width,
                                pad = left_margin, adjust_width = False)

    column_xs = [0] + np.cumsum(column_widths)[:-1].tolist()
    for i, (x, ax) in enumerate(zip(column_xs, colorbars)):
        curr = ax.get_position().bounds
        ax.set_position((1 - total_width + x + column_margin * i,) + curr[1:])

    plt.sca(prev_ax)



def apply_legend(style, *args, **kwargs):
    '''Apply a legend type based on a string name.

    The first parameter, style determines the legend type, while any other
    arguments given will be passed on directly to the legend function. The
    legend names resolve as:
    - `standard` --- ll.standard_legend
    - `covariate` --- ll.covariate_legend
    - `colorbar` or `vcolorbar` --- ll.vcolorbar_legend
    - `colorbars` or `vcolorbars` --- ll.vcolorbars_legend
    Note that any of the names can be preceded by 'll' and they will
    resolve in the same way.'''

    if 'covariate' in style or 'llcovariate' in style:
            legend_func = covariate_legend
    elif 'standard' in style or 'llstandard' in style:
        legend_func = standard_legend
    elif ('colorbar' in style or 'vcolorbar' in style or 
            'llcolorbar' in style or 'llvcolorbar' in style):
        legend_func = vcolorbar_legend
    elif ('colorbars' in style or 'vcolorbars' in style or
            'llcolorbars' in style or 'llvcolorbars' in style):
        legend_func = vcolorbars_legend
    else:
        raise ValueError("Unrecognized legend style: {}".format(style))
    legend_func(*args, **kwargs)

