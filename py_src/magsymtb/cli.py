from magsymtb.init_from_cif import main as init_main
from magsymtb.general_script import main as general_main
from magsymtb.run_diagonalization_band_plotting import main as diagonalization_main
from magsymtb.run_plot_band import main as plot_band_main


def run_init_from_cif():
    """This function is triggered by the 'mstb-init-from-cif' command."""
    init_main()

def run_general():
    general_main()

def run_diagonalization():
    """This function is triggered by the band plotting command."""
    diagonalization_main()

def run_plot_band():
    """This function is triggered by the plot band command."""
    plot_band_main()