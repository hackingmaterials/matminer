"""
Examples of formatting and using extra features of PlotlyFig
"""
import pprint
from matminer import PlotlyFig
from matminer.datasets.dataframe_loader import load_elastic_tensor

def plot_modes(api_key, username):
    """
    Demonstrate PlotlyFig plotting modes and show the easiest way to make
    adjustments.

    Offline mode - Set "mode" to "offline"
    Create a local html file. Note that offline mode in plotly disables LaTeX
    and some fonts on some systems by default. For the full-featured Plotly
    experience, please use the Plotly online mode.

    Static mode - Set "mode" to "static"
    Creates a single image file. Use height and width to specify the size of the
    image desired. api_key and username are required for static plotting mode.

    Online mode - Set "mode" to "online"
    Opens the figure in the Plotly online module.

    Notebook mode - Set "mode" to "notebook"
    Opens the figure in a Jupyter/IPython notebook. Not shown here, seen
    matminer_examples repository.

    Return mode - Pass "return_plot=True" into any plotting method
    Returns the figure as a 'bare-bones' dictionary. This can then be edited and
    passed into 'create_plot' of PlotlyFig or used directly with plotly.

    """

    if not api_key or not username:
        raise ValueError("Specify your Plotly api_key and username!")

    df = load_elastic_tensor()

    # First lets set uo our figure generally.
    pf = PlotlyFig(df, title='Elastic data', mode='offline', x_scale='log',
                   y_scale='log')

    # Lets plot offline (the default) first. An html file will be created.
    pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

    # Now lets plot again, but changing the filename and without opening.
    # We do this with the 'set_arguments' method.
    pf.set_arguments(show_offline_plot=False, filename="myplot.html")
    pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

    # Now lets create a static image.
    pf.set_arguments(mode='static',
                     api_key=api_key,
                     username=username,
                     filename="my_PlotlyFig_plot.jpeg")
    pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')
    # You can change the size of the image with the 'height' and 'width'
    # arguments to set_arguments.

    # Now we will use the Plotly online interface.
    pf.set_arguments(mode='online')
    pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula')

    # Great! Lets get the JSON representation of the PlotlyFig template as a
    # python dictionary. We can do this without changing the plot mode. From
    # any plotting method, simply pass 'return_plot=True' to return the plot.
    fig = pf.xy([('poisson_ratio', 'elastic_anisotropy')], labels='formula',
                return_plot=True)
    print("Here's our returned figure!")
    pprint.pprint(fig)

    # Edit the figure and plot it with the current plot mode (online):
    fig['layout']['hoverlabel']['bgcolor'] = 'pink'
    fig['layout']['title'] = 'My Custom Elastic Data Figure'
    pf.create_plot(fig)


def formatting_example(api_key, username):
    """
    Demonstrate common and advanced formatting features of PlotlyFig.

    PlotlyFig provides a set of arguments which make setting up good
    looking Plotly templates quick(er) and easy(er).

    Most formatting options can be set through the initializer of PlotlyFig.
    These options will remain the same for all figures producted, but you can
    change some common formatting options after instantitating a PlotlyFig
    object using set_arguments.

    Chart-specific formatting options can be passed to plotting methods.
    """

    if not api_key or not username:
        raise ValueError("Specify your Plotly api_key and username!")

    df = load_elastic_tensor()

    pf = PlotlyFig(df=df,
                   api_key=api_key,
                   username=username,
                   mode='online',
                   title='Comparison of Bulk Modulus and Shear Modulus',
                   x_title='Shear modulus (GPa)',
                   y_title='Bulk modulus (GPa)',
                   colorbar_title='Poisson Ratio',
                   fontfamily='Raleway',
                   fontscale=0.75,
                   fontcolor='#283747',
                   ticksize=30,
                   colorscale="Reds",
                   hovercolor='white',
                   hoverinfo='text',
                   bgcolor='#F4F6F6',
                   margins=110,
                   pad=10)

    pf.xy(('G_VRH', 'K_VRH'), labels='material_id', colors='poisson_ratio')


    # We can also use LaTeX if we use Plotly online/static
    pf.set_arguments(title="$\\text{Origin of Poisson Ratio } \\nu $",
                     y_title='$K_{VRH} \\text{(GPa)}$',
                     x_title='$G_{VRH} \\text{(GPa)}$',
                     colorbar_title='$\\nu$')
    pf.xy(('G_VRH', 'K_VRH'), labels='material_id', colors='poisson_ratio')



if __name__ == "__main__":

    MY_PLOTLY_USERNAME = ""
    MY_PLOTLY_API_KEY = ""
    plot_modes(MY_PLOTLY_API_KEY, MY_PLOTLY_USERNAME)
    formatting_example(MY_PLOTLY_API_KEY, MY_PLOTLY_USERNAME)