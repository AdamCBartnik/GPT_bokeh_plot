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


def gpt_plot(gpt_data_input, var1, var2, units=None, p=None, show_plot=True, format_input_data=True, **params):
    
    if (format_input_data):
        gpt_data = copy.deepcopy(gpt_data_input)
        for tout in gpt_data.tout:
            tout.drift_to_z()
    else:
        # Assume that this is already a copy of the data, and touts have been drift_to_z() already
        gpt_data = gpt_data_input
    
    p = make_default_plot(p, plot_width=400, plot_height=300, **params)
    line_colors = palettes.Set1[8]
        
    (x, x_units, x_scale) = scale_and_get_units(gpt_data.stat(var1, 'tout'), gpt_data.stat_units(var1).unitSymbol)
    screen_x = gpt_data.stat(var1, 'screen') / x_scale
    
    if (not isinstance(var2, list)):
        var2 = [var2]

    if ('n_slices' in params):
        for p in gpt_data.particles:
            p.n_slices = params['n_slices']
    if ('slice_key' in params):
        for p in gpt_data.particles:
            p.slice_key = params['slice_key']
    
    # Combine all y data into single array to find good units
    all_y = np.array([])
    all_y_base_units = gpt_data.stat_units(var2[0]).unitSymbol
    for var in var2:
        if (gpt_data.stat_units(var).unitSymbol != all_y_base_units):
            raise ValueError('Plotting data with different units not allowed.')
        all_y = np.concatenate((all_y, gpt_data.stat(var)))  # touts and screens for unit choices

    
    # In the case of emittance, use 2*median(y) as a the default scale, to avoid solenoid growth dominating the choice of scale
    use_median_y_strs = ['norm', 'slice']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        (_, y_units, y_scale) = scale_and_get_units(2.0*np.median(all_y), all_y_base_units)
        all_y = all_y / y_scale
    else:
        (all_y, y_units, y_scale) = scale_and_get_units(all_y, all_y_base_units)
    
    # Finally, actually plot the data
    for i, var in enumerate(var2):
        y = gpt_data.stat(var, 'tout') / y_scale
        p.line(x, y, line_width=2, color=line_colors[i], legend_label=f'{format_label(var)}')

        screen_y = gpt_data.stat(var, 'screen') / y_scale
        p.scatter(screen_x, screen_y, marker="circle", color=line_colors[i], size=8, legend_label=f'Screen: {format_label(var)}')
        
    # Axes labels
    ylabel_str = get_y_label(var2)
    p.xaxis.axis_label = f"{format_label(var1, use_base=True)} ({x_units})"
    p.yaxis.axis_label = f"{ylabel_str} ({y_units})"
    
    # Turn off or locate legend
    p.legend.click_policy='hide'
    p.legend.visible = False
    if('legend' in params):
        if (isinstance(params['legend'], str)):
            p.legend.visible = True
            p.legend.location = params['legend']
        if (isinstance(params['legend'], bool)):
            p.legend.visible = params['legend']
    
    
    # Cases where the y-axis should be forced to start at 0
    zero_y_strs = ['sigma_', 'charge', 'energy', 'slice', 'emit']
    if (any(any(substr in varstr for substr in zero_y_strs) for varstr in var2)):
        p.y_range.start = 0
    
    # Cases where the y-axis range should use the median, instead of the max (e.g. emittance plots)
    use_median_y_strs = ['norm_emit_x','norm_emit_y']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        p.y_range.end = 2.0*np.median(all_y)  
        
    if show_plot:
        show(p)
    

def gpt_plot_dist1d(pmd, var, plot_type='charge', units=None, p=None, p_table=None, show_plot=True, table_on=True, **params):
    plot_type = plot_type.lower()
    
    density_types = {'charge'}
    is_density = False
    if (plot_type in density_types):
        is_density = True
        
    positive_types = {'charge', 'norm_emit', 'sigma', 'slice'}
    is_positive = False
    if any([d in plot_type for d in positive_types]):
        is_positive = True
        
    min_particles = 1
    needs_many_particles_types = {'norm_emit', 'sigma'}
    if any([d in plot_type for d in positive_types]):
        min_particles = 5
    
    p = make_default_plot(p, plot_width=400, plot_height=300, **params)
    
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    pmd = postprocess_screen(pmd, **params)
            
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50

    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
        
    subtract_mean = False
    if (var in ['x','y','z','t']):
        subtract_mean = True
    (x, x_units, x_scale, mean_x, mean_x_units, mean_x_scale) = scale_mean_and_get_units(getattr(pmd, var), pmd.units(var).unitSymbol, subtract_mean=subtract_mean, weights=q)
           
    d_list, edges, density_norm = divide_particles(pmd, nbins=nbins, key=var)
    density_norm = density_norm*x_scale
    if (subtract_mean==True):
        edges = edges - mean_x*mean_x_scale
    edges = edges/x_scale
    
    plot_type_base_units = pmd.units(plot_type).unitSymbol
    _, plot_type_scale, plot_type_prefix = nicer_array(pmd[plot_type])
    plot_type_units = check_mu(plot_type_prefix)+plot_type_base_units
    norm = 1.0/plot_type_scale
    if (is_density):
        norm = norm*density_norm
        
    weights = np.array([0.0 for d in d_list])
    hist = np.array([0.0 for d in d_list])
    for d_i, d in enumerate(d_list):
        if (d.n_particle >= min_particles):
            hist[d_i] = d[plot_type]*norm
            weights[d_i] = d['charge']
    weights = weights/np.sum(weights)
    avg_hist = np.sum(hist*weights)
        
    edges, hist = duplicate_points_for_hist_plot(edges, hist)
    edges, hist = pad_data_with_zeros(edges, hist)
    
    p.line(edges, hist, line_width=2)
    
    p.xaxis.axis_label = f"{format_label(var)} ({x_units})"
    plot_type_label = get_y_label([plot_type])
    
    if (is_density):
        y_axis_label=f"{plot_type_label} density ({plot_type_units}/{x_units})"
    else:
        y_axis_label=f"{plot_type_label} ({plot_type_units})"
    
    p.yaxis.axis_label = y_axis_label
        
    if (is_positive):
        p.y_range.start = 0
    
    stdx = std_weights(x,q)
            
    if(table_on):
        var_label = format_label(var, remove_underscore=False, add_underscore=False)
        plot_type_label = format_label(plot_type, remove_underscore=False, add_underscore=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Total charge', col2=f'{q_total:G}', col3=f'{q_units}')
        if (not is_density):
            data = add_row(data, col1=f'Mean {plot_type_label}', col2=f'{avg_hist:G}', col3=f'{plot_type_units}')
        data = add_row(data, col1=f'Mean {var_label}', col2=f'{mean_x:G}', col3=f'{mean_x_units}')
        data = add_row(data, col1=f'σ_{var_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        p_table = make_parameter_table(p_table, data, headers, table_width=300, table_height=p.plot_height)
    
    if show_plot:
        if (p_table is not None):
            show(row_layout(p, p_table))
        else:
            show(p)
            


def gpt_plot_dist2d(pmd, var1, var2, ptype='hist2d', units=None, p=None, show_plot=True, table_on=True, **params):

    p = make_default_plot(p, plot_width=500, plot_height=400, tooltips=False, **params)
    p.grid.visible = False 
        
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    pmd = postprocess_screen(pmd, **params)
        
    if('axis' in params and params['axis']=='equal'):
        p.match_aspect = True
    
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
    
    (x, x_units, x_scale, avgx, avgx_units, avgx_scale) = scale_mean_and_get_units(getattr(pmd, var1), pmd.units(var1).unitSymbol, subtract_mean=True, weights=q)
    (y, y_units, y_scale, avgy, avgy_units, avgy_scale) = scale_mean_and_get_units(getattr(pmd, var2), pmd.units(var2).unitSymbol, subtract_mean=True, weights=q)
                
    if(ptype=="scatter"):
        color_var = 'density'
        if ('color_var' in params):
            color_var = params['color_var']
        scatter_color(p, pmd, x, y, color_var=color_var, bins=nbins, weights=q)
    if(ptype=="hist2d"):
        hist2d(p, x, y, bins=nbins, weights=q, colormap=colormap)
        
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
    
    p_table = None
    if(table_on):
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
        p_table = make_parameter_table(p_table, data, headers, table_width=300, table_height=p.plot_height)
    
    
    if show_plot:
        if (p_table is not None):
            show(row_layout(p, p_table))
        else:
            show(p)
            
        
    
    
    