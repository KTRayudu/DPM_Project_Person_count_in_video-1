from bokeh.models import ColumnDataSource
from bokeh.plotting import figure


def plot_creator(my_dataframe):
    """
     Returns a plot created by bokeh from the dataframe provided
    """
    source = ColumnDataSource(my_dataframe)
    fig = figure(plot_height=600, plot_width=720, tooltips=[('Person in frame ','@Person Count'), ('Frame active ','@Activity Indicator')])
    fig.line(x="Frame Number", y="Person Count", source=source, legend_label='Person Count')
    fig.line(x="Frame Number", y="Activity Indicator", source=source, color='red', legend_label='Activity Indicator')
    fig.legend.location = "top_left"
    fig.legend.title = "Objects Identified"
    fig.legend.background_fill_color = "navy"
    fig.legend.background_fill_alpha = 0.2
    fig.xaxis.axis_label = "Frame Number"
    fig.yaxis.axis_label = "Video Statistics"
    return fig

