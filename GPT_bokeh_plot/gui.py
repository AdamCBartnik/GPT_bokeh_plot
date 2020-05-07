from bokeh.plotting import figure, show
from bokeh.models import Panel, Tabs, Div, TextInput, Select
from bokeh.models import Range, Title, Grid, LinearAxis, Plot, Rect, Slider, CheckboxGroup
from bokeh.layouts import row
from bokeh.layouts import column
import numpy as np
from gpt.gpt import GPT as GPT
from .plot import gpt_plot, gpt_plot_dist1d, gpt_plot_dist2d
from bokeh.io.notebook import show_app, curstate
from bokeh.themes import Theme
import yaml
import copy
from .tools import make_default_plot, get_trend_vars, get_dist_plot_type, convert_gpt_data
from .ParticleGroupExtension import ParticleGroupExtension
from functools import partial

def gpt_plot_gui(gpt_data_input, primary_notebook_port=8888, secondary_gui_port=8889):
    notebook_address = f"http://localhost:{primary_notebook_port}"
    
    gpt_data = convert_gpt_data(gpt_data_input)
    
    screen_z_list = gpt_data.stat('mean_z', 'screen').tolist()
    screen_z_str_list = [f'{z:.3f}' for z in screen_z_list]
    
    label_width=150
    widget_width=150
    plot_area_width = 500
    
    text_input_params = {}
    
    def main_gui_fnc(doc):  
        
        # Make widgets
        plottype_list = ['Trends', '1D Distribution', '2D Distribution']
        plottype_dropdown = Select(title='Plot Type', value=plottype_list[0], options=plottype_list)
        
        trend_x_list = ['z', 't']
        trend_y_list = ['Beam Size', 'Bunch Length', 'Emittance (x,y)', 'Emittance (4D)', 'Slice emit. (x,y)', 'Slice emit. (4D)', 'Charge', 'Energy', 'Trajectory']
        trend_x_dropdown = Select(title='X axis', options=trend_x_list, value=trend_x_list[0])
        trend_y_dropdown = Select(title='Y axis', options=trend_y_list, value=trend_y_list[0])

        dist_list = ['t','x','y','px','py','pz']
        trend_slice_var_dropdown = Select(title='Slice variable', options=dist_list, value=dist_list[0])
        trend_slice_nslices_text = make_text_input(text_input_params, 'Number of slices', min=5, max=500, value=50, step=1)
        
        screen_z_dropdown = Select(title='Screen z (m)', options=screen_z_str_list, value=screen_z_str_list[0])
        dist_x_1d_dropdown = Select(title='X axis', options=dist_list, value=dist_list[0])
        dist_type_1d_list = ['Charge Density', 'Emittance X', 'Emittance Y', 'Emittance 4D', 'Sigma X', 'Sigma Y']
        dist_type_1d_dropdown = Select(title='Y axis', options=dist_type_1d_list, value=dist_type_1d_list[0])
        nbin_1d_text = make_text_input(text_input_params, 'Histogram bins', min=5, max=500, value=50, step=1)
                
        screen_z_dropdown_copy = Select(title='Screen z (m)', options=screen_z_str_list, value=screen_z_str_list[0])
        screen_z_dropdown_copy.js_link('value', screen_z_dropdown, 'value')
        screen_z_dropdown.js_link('value', screen_z_dropdown_copy, 'value')
        
        dist_x_dropdown = Select(title='X axis', options=dist_list, value=dist_list[1])
        dist_y_dropdown = Select(title='Y axis', options=dist_list, value=dist_list[2])   
        dist_2d_type_list = ['Scatter', 'Histogram']
        dist2d_type_dropdown = Select(title='Plot method', options=dist_2d_type_list, value=dist_2d_type_list[1])
        scatter_color_list = ['density','t','x','y','r','px','py','pz','pr']
        scatter_color_dropdown = Select(title='Scatter color variable', options=scatter_color_list, value=scatter_color_list[0])
        axis_equal_checkbox = CheckboxGroup(labels=['Enabled'], active=[])
        nbin_x_text = make_text_input(text_input_params, 'Histogram bins, X', min=5, max=500, value=50, step=1)
        nbin_y_text = make_text_input(text_input_params, 'Histogram bins, Y', min=5, max=500, value=50, step=1)
                    
        cyl_copies_checkbox = CheckboxGroup(labels=['Enabled'], active=[])
        cyl_copies_text = make_text_input(text_input_params, 'Number of copies', min=5, max=500, value=50, step=1)

        remove_correlation_checkbox = CheckboxGroup(labels=['Enabled'], active=[])
        remove_correlation_n_text = make_text_input(text_input_params, 'Max polynomial power', min=0, max=10, value=1, step=1)
        remove_correlation_var1_dropdown = Select(title='Independent var (x)', options=dist_list, value=dist_list[1])
        remove_correlation_var2_dropdown = Select(title='Dependent var (y)', options=dist_list, value=dist_list[3])

        take_slice_checkbox = CheckboxGroup(labels=['Enabled'], active=[])
        take_slice_var_dropdown = Select(title='Slice variable', options=dist_list, value=dist_list[0])
        take_slice_nslices_text = make_text_input(text_input_params, 'Number of slices', min=5, max=500, value=50, step=1)
        take_slice_index_text = make_text_input(text_input_params, 'Slice index', min=0, max=int(take_slice_nslices_text.value)-1, value=0, step=1)                      
                     
        trends_tab = column(
            add_label(trend_x_dropdown, widget_width),
            add_label(trend_y_dropdown, widget_width),
            add_label(trend_slice_var_dropdown, widget_width),
            add_label(trend_slice_nslices_text, widget_width)
        )
        
        dist_1d_tab = column(
            add_label(screen_z_dropdown, widget_width),
            add_label(dist_x_1d_dropdown, widget_width),
            add_label(dist_type_1d_dropdown, widget_width),
            add_label(nbin_1d_text, widget_width)
        )
        
        dist_2d_tab = column(
            add_label(screen_z_dropdown_copy, widget_width),
            add_label(dist2d_type_dropdown, widget_width),
            add_label(scatter_color_dropdown, widget_width),
            add_label(dist_x_dropdown, widget_width),
            add_label(dist_y_dropdown, widget_width),
            add_label(axis_equal_checkbox, widget_width, label='Equal scale axes'),
            add_label(nbin_x_text, widget_width),
            add_label(nbin_y_text, widget_width)
        )
                
        postprocess_tab = column(
            add_label(cyl_copies_checkbox, widget_width, label='Cylindrical copies'),
            add_label(cyl_copies_text, widget_width),
            add_label(remove_correlation_checkbox, widget_width, label='Remove Correlation'),
            add_label(remove_correlation_n_text, widget_width),
            add_label(remove_correlation_var1_dropdown, widget_width),
            add_label(remove_correlation_var2_dropdown, widget_width),
            add_label(take_slice_checkbox, widget_width, label='Take slice of data'),
            add_label(take_slice_var_dropdown, widget_width),
            add_label(take_slice_index_text, widget_width),
            add_label(take_slice_nslices_text, widget_width)
        )
                             
        tab1 = Panel(child=trends_tab, title='Trends')
        tab2 = Panel(child=dist_1d_tab, title='1D Dist.')
        tab3 = Panel(child=dist_2d_tab, title='2D Dist.')
        tab4 = Panel(child=postprocess_tab, title='Postprocess')
            
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
            
        main_panel = column(
            row(add_label(plottype_dropdown, widget_width)),
            tabs
        )
                   
        # Main plotting function
        def create_plot():
            #Get current GUI settings
            plottype = plottype_dropdown.value.lower()
            trend_x = 'mean_'+trend_x_dropdown.value.lower()
            trend_y = get_trend_vars(trend_y_dropdown.value.lower())
            trend_slice_var = trend_slice_var_dropdown.value
            trend_slice_nslices = int(trend_slice_nslices_text.value) #constrain_text_input(trend_slice_nslices_text, text_input_params)
            
            screen_z = float(screen_z_dropdown.value)
            dist_x_1d = dist_x_1d_dropdown.value
            dist_y_1d = dist_type_1d_dropdown.value.lower()
            nbins_1d = int(nbin_1d_text.value)  # constrain_text_input(nbin_1d_text, text_input_params)
            
            dist_x = dist_x_dropdown.value
            dist_y = dist_y_dropdown.value
            ptype = dist2d_type_dropdown.value.lower()
            
            scatter_color_var = scatter_color_dropdown.value.lower()
            is_trend = (plottype=='trends')
            is_slice_trend = any(['slice' in yy for yy in trend_y])
            is_dist1d = (plottype=='1d distribution')
            is_dist2d = (plottype=='2d distribution')
            nbins = [int(nbin_x_text.value), int(nbin_y_text.value)]
            axis_equal = (0 in axis_equal_checkbox.active)
                  
            cyl_copies_on = (0 in cyl_copies_checkbox.active) and (plottype!='trends')
            cyl_copies = int(cyl_copies_text.value)
            
            remove_correlation = (0 in remove_correlation_checkbox.active) and (plottype!='trends')
            remove_correlation_n = int(remove_correlation_n_text.value)
            remove_correlation_var1 = remove_correlation_var1_dropdown.value
            remove_correlation_var2 = remove_correlation_var2_dropdown.value
            
            take_slice = (0 in take_slice_checkbox.active) and (plottype!='trends')
            take_slice_var = take_slice_var_dropdown.value
            take_slice_nslices = int(take_slice_nslices_text.value)
            text_input_params[take_slice_index_text.id]['max'] = take_slice_nslices-1
            take_slice_index = int(take_slice_index_text.value)
            
            # Disable widgets
            trend_x_dropdown.disabled = not is_trend
            trend_y_dropdown.disabled = not is_trend
            trend_slice_var_dropdown.disabled = not is_slice_trend
            trend_slice_nslices_text.disabled = not is_slice_trend
            
            screen_z_dropdown.disabled = not is_dist1d           
            dist_x_1d_dropdown.disabled = not is_dist1d
            dist_type_1d_dropdown.disabled = not is_dist1d
            nbin_1d_text.disabled = not is_dist1d
            
            screen_z_dropdown_copy.disabled = not is_dist2d
            dist2d_type_dropdown.disabled = not is_dist2d
            scatter_color_dropdown.disabled = not (is_dist2d and ptype == 'scatter')
            dist_x_dropdown.disabled = not is_dist2d
            dist_y_dropdown.disabled = not is_dist2d
            axis_equal_checkbox.disabled = not is_dist2d
            nbin_x_text.disabled = not is_dist2d
            nbin_y_text.disabled = not is_dist2d
                        
            if (is_trend):
                params = {}
                if (is_slice_trend):
                    params['slice_key'] = trend_slice_var
                    params['n_slices'] = trend_slice_nslices
                p = gpt_plot(gpt_data, trend_x, trend_y, show_plot=False, format_input_data=False, **params)
            
            if (is_dist1d):
                ptype_1d = get_dist_plot_type(dist_y_1d)
                params={}
                if (cyl_copies_on):
                    params['cylindrical_copies'] = cyl_copies
                if (remove_correlation):
                    params['remove_correlation'] = (remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
                if (take_slice):
                    params['take_slice'] = (take_slice_var, take_slice_index, take_slice_nslices)
                p = gpt_plot_dist1d(gpt_data, dist_x_1d, screen_z=screen_z, plot_type=ptype_1d, nbins=nbins_1d, 
                                show_plot=False, format_input_data=False, **params)
            if (is_dist2d):
                params={}
                params['color_var'] = scatter_color_var
                if (axis_equal):
                    params['axis'] = 'equal'
                if (cyl_copies_on):
                    params['cylindrical_copies'] = cyl_copies
                if (remove_correlation):
                    params['remove_correlation'] = (remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
                if (take_slice):
                    params['take_slice'] = (take_slice_var, take_slice_index, take_slice_nslices)
                p = gpt_plot_dist2d(gpt_data, dist_x, dist_y, screen_z=screen_z, plot_type=ptype, nbins=nbins, 
                                show_plot=False, format_input_data=False, **params)
                
            p.width = plot_area_width
            return p     
            
        gui = row(main_panel, create_plot())
        doc.add_root(gui)
            
        #callback functions
        def just_redraw(attr, old, new):
             gui.children[1] = create_plot()
                    
        def change_tab(attr, old, new):
            new_index = plottype_list.index(new)
            if (tabs.active < 3):
                tabs.active = new_index
            gui.children[1] = create_plot()
                                        
        # Assign callbacks
        plottype_dropdown.on_change('value', change_tab)
        trend_x_dropdown.on_change('value', just_redraw)
        trend_y_dropdown.on_change('value', just_redraw)
        trend_slice_var_dropdown.on_change('value', just_redraw)
        trend_slice_nslices_text.on_change('value', just_redraw)
                         
        screen_z_dropdown.on_change('value', just_redraw)
        dist_x_1d_dropdown.on_change('value', just_redraw)
        dist_type_1d_dropdown.on_change('value', just_redraw)
        nbin_1d_text.on_change('value', just_redraw)
        
        dist_x_dropdown.on_change('value', just_redraw)
        dist_y_dropdown.on_change('value', just_redraw)
        dist2d_type_dropdown.on_change('value', just_redraw)
        scatter_color_dropdown.on_change('value', just_redraw)
        nbin_x_text.on_change('value', just_redraw)
        nbin_y_text.on_change('value', just_redraw)
        axis_equal_checkbox.on_change('active', just_redraw)
        
        cyl_copies_checkbox.on_change('active', just_redraw)
        cyl_copies_text.on_change('value', just_redraw)
        remove_correlation_checkbox.on_change('active', just_redraw)
        remove_correlation_n_text.on_change('value', just_redraw)
        remove_correlation_var1_dropdown.on_change('value', just_redraw)
        remove_correlation_var2_dropdown.on_change('value', just_redraw)
        take_slice_checkbox.on_change('active', just_redraw)
        take_slice_var_dropdown.on_change('value', just_redraw)
        take_slice_index_text.on_change('value', just_redraw)
        take_slice_nslices_text.on_change('value', just_redraw)
                     
    # Run the GUI
    show_app(main_gui_fnc, curstate(), notebook_address, port=secondary_gui_port)
    
    

    
    def constrain_text_input(widget, params_list):
        params = params_list[widget.id]
        new = float(widget.value)
        if ('min' in params):
            if (new < float(params['min'])):
                new = float(params['min'])
        if ('max' in params):
            if (new > float(params['max'])):
                new = float(params['max'])
        if ('step' in params and 'min' in params):
            new = float(params['min']) + float(params['step'])*np.round((new - float(params['min']))/float(params['step']))
            if (params['step'] == 1):
                new = int(new)
        return new
    
    def make_text_input(params, title, min=0, max=1000, value=10, step=1):
        p = {}
        p['min'] = min
        p['max'] = max
        p['step'] = step
        widget = TextInput(value=f'{value}', title=title)
        params[widget.id] = p
        return widget
    
    
    def add_label(w, widget_width, label=None):
        label_width = 150
        if (label==None):
            label = w.title
            w.title = ''
        return row(Div(text=label, width=label_width), w)