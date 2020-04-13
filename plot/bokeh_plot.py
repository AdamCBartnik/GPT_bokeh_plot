from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Title, ColumnDataSource, Grid, LinearAxis, Plot, Rect
from bokeh.colors import RGB
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot
from bokeh.layouts import row as row_layout
from bokeh import palettes  as palettes 
import numpy as np
import matplotlib as mpl
from distgen.tools import *
from gpt.gpt import GPT as GPT
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import multiply_units
from operator import itemgetter 
from .tools import *
from .nicer_units import *
from pmd_beamphysics.units import c_light, e_charge


def bokeh_plot(gpt_data, var1, var2, units=None, p=None, show_plot=True, **params):
    
    p = make_default_plot(p, plot_width=400, plot_height=300, **params)
        
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50

    (x, x_units, x_scale) = scale_and_get_units(gpt_data.stat(var1, 'tout'), gpt_data.stat_units(var1).unitSymbol)
    screen_x = gpt_data.stat(var1, 'screen') / x_scale
    
    if (not isinstance(var2, list)):
        var2 = [var2]
    
    all_y = np.array([])
    all_y_base_units = gpt_data.stat_units(var2[0]).unitSymbol

    for var in var2:
        if (gpt_data.stat_units(var).unitSymbol != all_y_base_units):
            raise ValueError('Plotting data with different units not allowed.')
        all_y = np.concatenate((all_y, gpt_data.stat(var)))  # touts and screens for unit choices

    (all_y, y_units, y_scale) = scale_and_get_units(all_y, all_y_base_units)

    line_colors = palettes.Set1[8]
    
    for i, var in enumerate(var2):
        y = gpt_data.stat(var, 'tout') / y_scale
        p.line(x, y, line_width=2, color=line_colors[i], legend_label=f'{format_label(var)}')

        screen_y = gpt_data.stat(var, 'screen') / y_scale
        p.scatter(screen_x, screen_y, marker="circle", color=line_colors[i], size=8, legend_label=f'Screen: {format_label(var)}')
        
    ylabel_str = get_y_label(var2)
                
    p.xaxis.axis_label = f"{format_label(var1, use_base=True)} ({x_units})"
    p.yaxis.axis_label = f"{ylabel_str} ({y_units})"
    
    zero_y_strs = ['sigma_', 'norm_', 'charge']
    if (any(any(substr in varstr for substr in zero_y_strs) for varstr in var2)):
        p.y_range.start = 0       
    
    p.legend.click_policy='hide'
    p.legend.visible = False
    if('legend' in params):
        if (isinstance(params['legend'], str)):
            p.legend.visible = True
            p.legend.location = params['legend']
        if (isinstance(params['legend'], bool)):
            p.legend.visible = params['legend']
        
    if show_plot:
        show(p)
    
    return p



def bokeh_plot_dist1d(pmd, var, units=None, p=None, show_plot=True, **params):
    
    p = make_default_plot(p, plot_width=400, plot_height=300, **params)
    
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
            
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50

    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
        
    subtract_mean = True
    if (var == 'r'):
        subtract_mean = False
    (x, x_units, x_scale, mean_x, mean_x_units, mean_x_scale) = scale_mean_and_get_units(getattr(pmd, var), pmd.units(var).unitSymbol, subtract_mean=subtract_mean, weights=q)
           
    if (var == 'r'):
        hist, edges = radial_histogram_no_units(x, weights=q, nbins=nbins)
    else:
        hist, edges = np.histogram(x,bins=nbins,weights=q,density=True)
        hist *= q_total;

    edges, hist = duplicate_points_for_hist_plot(edges, hist)
    
    if (var != 'r'):
        edges, hist = pad_data_with_zeros(edges, hist)

    p.line(edges, hist, line_width=2)
    p.xaxis.axis_label = f"{format_label(var)} ({x_units})"
    
    if (var == 'r'):
        p.yaxis.axis_label = f"Charge density ({q_units}/{x_units}²)"
    else:
        p.yaxis.axis_label = f"Charge density ({q_units}/{x_units})"
    
    stdx = std_weights(x,q)
    
    if('title_on' in params and params['title_on']):
        avgx_str = f"{mean_x:G} {mean_x_units}"
        stdx_str = f"{stdx:G} {x_units}"
        qb_str = f"{q_total:G} {q_units}"
        
        title_text = f'<{labels[var]}> = {avgx_str}, σ_{labels[var]} = {stdx_str}'
        sub_title_text = f'Total charge = {qb_str}'
        p.add_layout(Title(text=title_text, text_font_style="normal", align="center"), 'above')
        p.add_layout(Title(text=sub_title_text, text_font_style="normal", align="center"), 'above')
        
    p_table = None
    if(('table_on' in params and params['table_on']) or ('table_on' not in params)):
        var_label = format_label(var, remove_underscore=False, add_underscore=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Total charge', col2=f'{q_total:G}', col3=f'{q_units}')
        data = add_row(data, col1=f'Mean {var_label}', col2=f'{mean_x:G}', col3=f'{mean_x_units}')
        data = add_row(data, col1=f'σ_{var_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        p_table = make_parameter_table(data, headers, table_width=300, table_height=p.plot_height)
    
    
    if show_plot:
        if (p_table is not None):
            show(row_layout(p, p_table))
        else:
            show(p)
            
    
    return p



def bokeh_plot_dist2d(pmd, var1, var2, ptype='hist2d', units=None, p=None, show_plot=True, **params):

    p = make_default_plot(p, plot_width=500, plot_height=400, tooltips=False, **params)
         
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
        
    p.grid.visible = False
    
    if('axis' in params and params['axis']=='equal'):
        p.match_aspect = True
    
    is_radial_var = [False, False]
    if (var1 == 'r'):
        is_radial_var[0] = True
    if (var2 == 'r'):
        is_radial_var[1] = True
        
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
        
    if (not isinstance(nbins, list)):
        nbins = [nbins, nbins]
        
    if ('colormap' in params):
        colormap = mpl.cm.get_cmap(params[colormap])
    else:
        colormap = mpl.cm.get_cmap('jet') 
       
    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    (x, x_units, x_scale, avgx, avgx_units, avgx_scale) = scale_mean_and_get_units(getattr(pmd, var1), pmd.units(var1).unitSymbol, subtract_mean= not is_radial_var[0], weights=q)
    (y, y_units, y_scale, avgy, avgy_units, avgy_scale) = scale_mean_and_get_units(getattr(pmd, var2), pmd.units(var2).unitSymbol, subtract_mean= not is_radial_var[1], weights=q)
                
    if(ptype=="scatter"):
        p.scatter(x, y, line_color=None)
    if(ptype=="hist2d"):
        hist2d(p, x, y, bins=nbins, weights=q, colormap=colormap, is_radial_var=is_radial_var)
    if(ptype=="scatter_hist2d"):
        scatter_hist2d(p, x, y, bins=nbins, weights=q, colormap=colormap, is_radial_var=is_radial_var)
        
    p.xaxis.axis_label = f"{format_label(var1)} ({x_units})"
    p.yaxis.axis_label = f"{format_label(var2)} ({y_units})"
             
    stdx = std_weights(x,q)
    stdy = std_weights(y,q)
    corxy = corr_weights(x,y,q)
    if (x_units == y_units):
        corxy_units = f'{x_units}²'
    else:
        corxy_units = f'{x_units}·{y_units}'
    
    show_emit = False
    if ((var1 == 'x' and var2 == 'px') or (var1 == 'y' and var2 == 'py')):
        show_emit = True
        factor = c_light**2 /e_charge # kg -> eV
        particle_mass = 9.10938356e-31  # kg
        emitxy = (x_scale*y_scale/factor/particle_mass)*np.sqrt(stdx**2 * stdy**2 - corxy**2)
        (emitxy, emitxy_units, emitxy_scale) = scale_and_get_units(emitxy, pmd.units(var1).unitSymbol)

    if('title_on' in params and params['title_on']):
        avgx_str = f"{avgx:G} {avgx_units}"
        stdx_str = f"{stdx:G} {x_units}"
        avgy_str = f"{avgy:G} {avgy_units}"
        stdy_str = f"{stdy:G} {y_units}"
        qb_str = f"{q_total:G} {q_units}"
        
        title_text = f'<{labels[var1]}> = {avgx_str}, σ_{labels[var1]} = {stdx_str}'
        title_text_2 = f'<{labels[var2]}> = {avgy_str}, σ_{labels[var2]} = {stdy_str}'

        p.add_layout(Title(text=title_text_2, text_font_style="normal", align="center"), 'above')
        p.add_layout(Title(text=title_text, text_font_style="normal", align="center"), 'above')
    
    p_table = None
    if(('table_on' in params and params['table_on']) or ('table_on' not in params)):
        var1_label = format_label(var1, remove_underscore=False, add_underscore=False)
        var2_label = format_label(var2, remove_underscore=False, add_underscore=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Mean {var1_label}', col2=f'{avgx:G}', col3=f'{avgx_units}')
        data = add_row(data, col1=f'Mean {var2_label}', col2=f'{avgy:G}', col3=f'{avgy_units}')
        data = add_row(data, col1=f'σ_{var1_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        data = add_row(data, col1=f'σ_{var2_label}', col2=f'{stdy:G}', col3=f'{y_units}')
        data = add_row(data, col1=f'Corr({var1_label}, {var2_label})', col2=f'{corxy:G}', col3=f'{corxy_units}')
        if (show_emit):
            data = add_row(data, col1=f'ε_{var1_label}', col2=f'{emitxy:G}', col3=f'{emitxy_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        p_table = make_parameter_table(data, headers, table_width=300, table_height=p.plot_height)
    
    
    if show_plot:
        if (p_table is not None):
            show(row_layout(p, p_table))
        else:
            show(p)
            
    return p
        
    
    
    