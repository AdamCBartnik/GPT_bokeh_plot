import numpy as np
import copy
from .tools import make_default_plot, get_trend_vars, get_dist_plot_type, convert_gpt_data, make_default_plot
from .plot import gpt_plot, gpt_plot_dist1d, gpt_plot_dist2d
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row as row_layout
from bokeh.models import DataTable
import ipywidgets as widgets

def gpt_plot_gui(gpt_data_input):
    gpt_data = convert_gpt_data(gpt_data_input)
    p = make_default_plot(None, plot_width=500, plot_height=400)
    p_table=DataTable()
    
    screen_z_list = gpt_data.stat('mean_z', 'screen').tolist()
    
    tab_panel = widgets.Tab()
    
    # Layouts
    layout_200px = widgets.Layout(width='200px',height='30px')
    layout_150px = widgets.Layout(width='150px',height='30px')
    layout_100px = widgets.Layout(width='100px',height='30px')
    layout_20px = widgets.Layout(width='20px',height='30px')
    label_layout = layout_150px
    
    # Make widgets
    plottype_list = ['Trends', '1D Distribution', '2D Distribution']
    plottype_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(plottype_list)], value=0)
    
    dist2d_type_dropdown = widgets.Dropdown(options=[('Scatter', 'scatter'), ('Histogram', 'histogram')], 
                                            value='histogram', layout=layout_150px)
    scatter_color = ['density','t','x','y','r','px','py','pz','pr']
    scatter_color_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(scatter_color)], value=0, layout=layout_150px)
    
    trend_x_list = ['z', 't']
    trend_y_list = ['Beam Size', 'Bunch Length', 'Emittance (x,y)', 'Emittance (4D)', 'Slice emit. (x,y)', 'Slice emit. (4D)', 'Charge', 'Energy', 'Trajectory']
    trend_x_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(trend_x_list)], value=0, layout=layout_150px)
    trend_y_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(trend_y_list)], value=0, layout=layout_150px)
    
    dist_list = ['t','x','y','r','px','py','pz','pr']
    trend_slice_var_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    trend_slice_nslices_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    
    dist_x_1d_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    dist_x_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=1, layout=layout_150px)
    dist_y_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=2, layout=layout_150px)
    
    dist_type_1d_list = ['Charge Density', 'Emittance X', 'Emittance Y', 'Emittance 4D', 'Sigma X', 'Sigma Y']
    dist_type_1d_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_type_1d_list)], value=0, layout=layout_150px)
    
    axis_equal_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    
    screen_z_dropdown = widgets.Dropdown(options=[(f'{z:.3f}', i) for (i,z) in enumerate(screen_z_list)], layout=layout_150px)
    
    nbin_1d_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    nbin_x_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    nbin_y_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    
    cyl_copies_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    cyl_copies_text = widgets.BoundedIntText(value=50, min=10, max=500, step=1, layout=layout_150px)
    
    remove_correlation_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    remove_correlation_n_text = widgets.BoundedIntText(value=1, min=0, max=10, step=1, layout=layout_150px)
    remove_correlation_var1_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    remove_correlation_var2_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=6, layout=layout_150px)
    
    take_slice_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    take_slice_var_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    take_slice_nslices_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    take_slice_index_text = widgets.BoundedIntText(value=0, min=0, max=take_slice_nslices_text.value-1, step=1, layout=layout_150px)
    
    trends_tab = widgets.VBox([
        widgets.HBox([widgets.Label('X axis', layout=label_layout), trend_x_dropdown]),
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), trend_y_dropdown]),
        widgets.HBox([widgets.Label('Slice variable', layout=label_layout), trend_slice_var_dropdown]),
        widgets.HBox([widgets.Label('Number of slices', layout=label_layout), trend_slice_nslices_text])
    ])
    
    dist_1d_tab = widgets.VBox([
        widgets.HBox([widgets.Label('Screen z (m)', layout=label_layout), screen_z_dropdown]),
        widgets.HBox([widgets.Label('X axis', layout=label_layout), dist_x_1d_dropdown]), 
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), dist_type_1d_dropdown]),
        widgets.HBox([widgets.Label('Histogram bins', layout=label_layout), nbin_1d_text])
    ])

    dist_2d_tab = widgets.VBox([
        widgets.HBox([widgets.Label('Screen z (m)', layout=label_layout), screen_z_dropdown]),
        widgets.HBox([widgets.Label('Plot method', layout=label_layout), dist2d_type_dropdown]),
        widgets.HBox([widgets.Label('Scatter color variable', layout=label_layout), scatter_color_dropdown]),
        widgets.HBox([widgets.Label('X axis', layout=label_layout), dist_x_dropdown]), 
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), dist_y_dropdown]),
        widgets.HBox([widgets.Label('Equal scale axes', layout=label_layout), axis_equal_checkbox]),
        widgets.HBox([widgets.Label('Histogram bins, X', layout=label_layout), nbin_x_text]),
        widgets.HBox([widgets.Label('Histogram bins, Y', layout=label_layout), nbin_y_text])
    ])
    
    postprocess_tab = widgets.VBox([
        widgets.HBox([widgets.Label('Cylindrical copies', layout=label_layout), cyl_copies_checkbox]),
        widgets.HBox([widgets.Label('Number of copies', layout=label_layout), cyl_copies_text]),
        widgets.HBox([widgets.Label('Remove Correlation', layout=label_layout), remove_correlation_checkbox]),
        widgets.HBox([widgets.Label('Max polynomial power', layout=label_layout), remove_correlation_n_text]),
        widgets.HBox([widgets.Label('Independent var (x)', layout=label_layout), remove_correlation_var1_dropdown]), 
        widgets.HBox([widgets.Label('Dependent var (y)', layout=label_layout), remove_correlation_var2_dropdown]),
        widgets.HBox([widgets.Label('Take slice of data', layout=label_layout), take_slice_checkbox]), 
        widgets.HBox([widgets.Label('Slice variable', layout=label_layout), take_slice_var_dropdown]), 
        widgets.HBox([widgets.Label('Slice index', layout=label_layout), take_slice_index_text]), 
        widgets.HBox([widgets.Label('Number of slices', layout=label_layout), take_slice_nslices_text])
    ], description='Postprocessing')
    
    tab_list = [trends_tab, dist_1d_tab, dist_2d_tab, postprocess_tab]
    tab_label_list = ['Trends', '1D Dist.', '2D Dist.', 'Postprocess']
    tab_panel.children = tab_list
    for i,t in enumerate(tab_label_list):
        tab_panel.set_title(i, t)
    
    tools_panel = widgets.VBox([
        widgets.HBox([widgets.Label('Plot Type', layout=label_layout), plottype_dropdown]),
        tab_panel
    ])
    
        
    def make_plot():   
        plottype = plottype_dropdown.label.lower()
        trend_x = trend_x_dropdown.label
        trend_y = trend_y_dropdown.label
        dist_x_1d = dist_x_1d_dropdown.label
        dist_y_1d = dist_type_1d_dropdown.label
        dist_x = dist_x_dropdown.label
        dist_y = dist_y_dropdown.label
        screen_z = screen_z_list[screen_z_dropdown.value]
        nbins_1d = nbin_1d_text.value
        nbins = [nbin_x_text.value, nbin_y_text.value]
        cyl_copies = cyl_copies_text.value
        cyl_copies_on = cyl_copies_checkbox.value and (plottype!='trends')
        cyl_copies_text.disabled = not cyl_copies_on
        ptype = dist2d_type_dropdown.value.lower()
        scatter_color_var = scatter_color_dropdown.label.lower()
        axis_equal = axis_equal_checkbox.value
        remove_correlation = remove_correlation_checkbox.value and (plottype!='trends')
        remove_correlation_n = remove_correlation_n_text.value
        remove_correlation_var1 = remove_correlation_var1_dropdown.label
        remove_correlation_var2 = remove_correlation_var2_dropdown.label
        take_slice = take_slice_checkbox.value and (plottype!='trends')
        take_slice_var = take_slice_var_dropdown.label
        take_slice_index = take_slice_index_text.value
        take_slice_nslices = take_slice_nslices_text.value
        take_slice_index_text.max = take_slice_nslices-1
        trend_slice_var = trend_slice_var_dropdown.label
        trend_slice_nslices = trend_slice_nslices_text.value
        
        is_trend = (plottype=='trends')
        is_dist1d = (plottype=='1d distribution')
        is_dist2d = (plottype=='2d distribution')
        
        is_slice_trend = ('slice' in trend_y.lower())
                
        trend_x_dropdown.disabled = not is_trend
        trend_y_dropdown.disabled = not is_trend
        dist_x_1d_dropdown.disabled = not is_dist1d
        dist_type_1d_dropdown.disabled = not is_dist1d
        dist_x_dropdown.disabled = not is_dist2d
        dist_y_dropdown.disabled = not is_dist2d
        nbin_1d_text.disabled = not is_dist1d
        nbin_x_text.disabled = not is_dist2d
        nbin_y_text.disabled = not is_dist2d
        screen_z_dropdown.disabled = not (is_dist1d or is_dist2d)
        cyl_copies_checkbox.disabled = not (is_dist1d or is_dist2d)
        dist2d_type_dropdown.disabled = not is_dist2d
        axis_equal_checkbox.disabled = not is_dist2d
        scatter_color_dropdown.disabled = not (is_dist2d and ptype == 'scatter')
        trend_slice_var_dropdown.disabled = not is_slice_trend
        trend_slice_nslices_text.disabled = not is_slice_trend
        
        if (is_trend):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 0
            var1 = 'mean_'+trend_x
            var2 = get_trend_vars(trend_y)
            params = {}
            if (is_slice_trend):
                params['slice_key'] = trend_slice_var
                params['n_slices'] = trend_slice_nslices
            gpt_plot(gpt_data, var1, var2, p=p, show_plot=False, format_input_data=False, **params)
            #figure_hbox.children += (gpt_plot(gpt_data, var1, var2, **params), )
        if (is_dist1d):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 1
            ptype_1d = get_dist_plot_type(dist_y_1d)
            params = {}
            if (cyl_copies_on):
                params['cylindrical_copies'] = cyl_copies
            if (remove_correlation):
                params['remove_correlation'] = (remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
            if (take_slice):
                params['take_slice'] = (take_slice_var, take_slice_index, take_slice_nslices)
            gpt_plot_dist1d(gpt_data, dist_x_1d, p=p, p_table=p_table, screen_z=screen_z, plot_type=ptype_1d, nbins=nbins_1d, 
                            show_plot=False, format_input_data=False, **params)
        if (is_dist2d):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 2
            params = {}
            params['color_var'] = scatter_color_var
            if (axis_equal):
                params['axis'] = 'equal'
            if (cyl_copies_on):
                params['cylindrical_copies'] = cyl_copies
            if (remove_correlation):
                params['remove_correlation'] = (remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
            if (take_slice):
                params['take_slice'] = (take_slice_var, take_slice_index, take_slice_nslices)
            gpt_plot_dist2d(gpt_data, dist_x, dist_y, p=p, p_table=p_table, screen_z=screen_z, plot_type=ptype, nbins=nbins, 
                            show_plot=False, format_input_data=False, **params)
            
            
    # Callback functions
    def remake_on_value_change(change):
        p.renderers = []
        make_plot()
        push_notebook()
                    
    # Register callbacks
    plottype_dropdown.observe(remake_on_value_change, names='value')
    trend_x_dropdown.observe(remake_on_value_change, names='value')
    trend_y_dropdown.observe(remake_on_value_change, names='value')
    dist_x_1d_dropdown.observe(remake_on_value_change, names='value')
    dist_x_dropdown.observe(remake_on_value_change, names='value')
    dist_y_dropdown.observe(remake_on_value_change, names='value')
    screen_z_dropdown.observe(remake_on_value_change, names='value')
    nbin_1d_text.observe(remake_on_value_change, names='value')
    nbin_x_text.observe(remake_on_value_change, names='value')
    nbin_y_text.observe(remake_on_value_change, names='value')
    cyl_copies_checkbox.observe(remake_on_value_change, names='value')
    cyl_copies_text.observe(remake_on_value_change, names='value')
    dist2d_type_dropdown.observe(remake_on_value_change, names='value')
    scatter_color_dropdown.observe(remake_on_value_change, names='value')
    axis_equal_checkbox.observe(remake_on_value_change, names='value')
    dist_type_1d_dropdown.observe(remake_on_value_change, names='value')
    remove_correlation_checkbox.observe(remake_on_value_change, names='value')
    remove_correlation_n_text.observe(remake_on_value_change, names='value')
    remove_correlation_var1_dropdown.observe(remake_on_value_change, names='value')
    remove_correlation_var2_dropdown.observe(remake_on_value_change, names='value')
    take_slice_checkbox.observe(remake_on_value_change, names='value')
    take_slice_var_dropdown.observe(remake_on_value_change, names='value')
    take_slice_index_text.observe(remake_on_value_change, names='value')
    take_slice_nslices_text.observe(remake_on_value_change, names='value')
    trend_slice_var_dropdown.observe(remake_on_value_change, names='value')
    trend_slice_nslices_text.observe(remake_on_value_change, names='value')
    
    make_plot()
    gui = tools_panel
        
    return (p,p_table,gui)

