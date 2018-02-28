"""
This script shows some basic examples of xy plot using figrecipes in matminer.

To see the examples plot_thermoelectrics and plot_expt_compt_band_gaps, make
sure to enter your Citrine API key or have it set inside CITRINE_KEY
environment variable. Also, when using MPDataRetrieval make sure your API key
for the Materials Project (https://materialsproject.org/open) is set inside
MAPI_KEY environment variable.
"""

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataframe_loader import load_elastic_tensor
from matminer.figrecipes.plot import PlotlyFig
from pymatgen import Composition

def plot_simple_xy():
    """
    Very simple xy plot with all default settings.
    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    pf = PlotlyFig(title="Basic Example", filename="basic.html")
    pf.xy(([1, 2, 3], [4, 5, 6]))


def plot_bulk_shear_moduli():
    """
    Very basic example of xy scatter plot of Voigt-Reuss-Hill (VRH) average
        bulk vs. shear modulus. Poisson ratio as marker colors make the
        distinction between materials with different bulk/shear modulus ratios
    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    df = load_elastic_tensor()
    pf = PlotlyFig(df,
                   y_title='Bulk Modulus (GPa)',
                   x_title='Shear Modulus (GPa)',
                   filename='bulk_shear_moduli')
    pf.xy(('G_VRH', 'K_VRH'),
          labels='material_id',
          colors='poisson_ratio',
          colorscale='Picnic')


def plot_thermoelectrics(citrine_api_key, limit=0):
    """
    Scatter plot of the properties of thermoelectric materials based on the data
        available in http://www.mrl.ucsb.edu:8080/datamine/thermoelectric.jsp
        The data is extracted via Citrine data retrieval tools. The dataset
        id on Citrine is 150557
    Args:
        citrine_api_key (str): Your Citrine API key for getting data. Don't have
            a Citrine account? Visit https://citrine.io/
        limit (int): limit the number of entries (0 means no limit)
    Returns:
        plotly plot in "offline" mode poped in the default browser.
    """
    cdr = CitrineDataRetrieval(api_key=citrine_api_key)
    cols = ['chemicalFormula', 'Electrical resistivity', 'Seebeck coefficient',
            'Thermal conductivity', 'Thermoelectric figure of merit (zT)']
    df_te = cdr.get_dataframe(data_type='experimental', data_set_id=150557,
                              show_columns=cols, max_results=limit
                              ).set_index('chemicalFormula').astype(float)
    df_te = df_te[(df_te['Electrical resistivity'] > 5e-4) & \
                  (df_te['Electrical resistivity'] < 0.1)]
    df_te = df_te[abs(df_te['Seebeck coefficient']) < 500].rename(
                columns={'Thermoelectric figure of merit (zT)': 'zT'})

    print(df_te.head())
    pf = PlotlyFig(df_te,
                   x_scale='log',
                   fontfamily='Times New Roman',
                   hovercolor='white',
                   x_title='Electrical Resistivity (cm/S)',
                   y_title='Seebeck Coefficient (uV/K)',
                   colorbar_title='Thermal Conductivity (W/m.K)',
                   filename='thermoelectrics.html')
    pf.xy(('Electrical resistivity', 'Seebeck coefficient'),
          labels=df_te.index,
          sizes='zT',
          colors='Thermal conductivity',
          color_range=[0, 5])


def plot_expt_compt_band_gaps(citrine_api_key, limit=0):
    """
    Pulls experimental band gaps from Citrine (w/o dataset limitations) and
        evaluate the DFT computed band gaps (data from materialsproject.org)
        in xy scatter plot. To compare the right values, we pick the computed
        band gaps calculated for a chemical formula that has the lowest energy
        above hull (the most stable structure).
    Args:
        citrine_api_key (str): Your Citrine API key for getting data. Don't have
            a Citrine account? Visit https://citrine.io/
        limit (int): limit the number of entries (0 means no limit)
    Returns:
        plotly plots in "offline" mode poped in the default browser.
    """

    # pull experimental band gaps from Citrine
    cdr = CitrineDataRetrieval(api_key=citrine_api_key)
    cols = ['chemicalFormula', 'Band gap']
    df_ct = cdr.get_dataframe(prop='band gap', data_type='experimental',
                              show_columns=cols, max_results=limit).rename(
        columns={'chemicalFormula': 'Formula', 'Band gap': 'Expt. gap'})
    df_ct = df_ct[df_ct['Formula'] != 'In1p1'] # p1 not recognized in Composition
    df_ct = df_ct.dropna() # null band gaps cause problem when plotting residuals
    df_ct['Formula'] = df_ct['Formula'].transform(
        lambda x: Composition(x).get_reduced_formula_and_factor()[0])

    # pull computational band gaps from the Materials Project
    df = MPDataRetrieval().get_dataframe(
        criteria={'pretty_formula': {'$in': list(df_ct['Formula'].values)}},
        properties=['pretty_formula', 'material_id', 'band_gap', 'e_above_hull'],
        index_mpid=False).rename(
        columns={'pretty_formula': 'Formula', 'band_gap': 'MP computed gap',
                 'material_id': 'mpid'})


    # pick the most stable structure
    df_mp = df.loc[df.groupby("Formula")["e_above_hull"].idxmin()]
    df_final = df_ct.merge(df_mp, on='Formula').drop(
                                    'e_above_hull', axis=1).set_index('mpid')
    pf = PlotlyFig(df_final, x_title='Experimental band gap (eV)',
                   y_title='Computed Band Gap (eV)',
                   filename='band_gaps')

    # computed vs. experimental band gap:
    pf.xy([
        ('Expt. gap', 'MP computed gap'),
        ([0, 12], [0, 12])
    ],
        lines=[{}, {'color': 'black', 'dash': 'dash'}],
        labels=df_final.index, modes=['markers', 'lines'],
        names=['Computed vs. expt.', 'Expt. gap'])

    # residual:
    residuals = df_final['MP computed gap']-df_final['Expt. gap'].astype(float)
    pf.set_arguments(x_title='Experimental band gap (eV)',
                    y_title='Residual (Computed - Expt.) Band Gap (eV)',
                    filename='band_gap_residuals')
    pf.xy(('Expt. gap', residuals), labels = df_final.index)


if __name__ == '__main__':
    plot_simple_xy()
    plot_bulk_shear_moduli()

    MY_CITRINE_API_KEY=""
    plot_thermoelectrics(MY_CITRINE_API_KEY, limit=0)
    plot_expt_compt_band_gaps(MY_CITRINE_API_KEY, limit=0)