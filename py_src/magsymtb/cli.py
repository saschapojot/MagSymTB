from magsymtb.init_from_cif import main as init_main
from magsymtb.general_script import main as general_main


def run_init_from_cif():
    """This function is triggered by the 'mstb-init-from-cif' command."""
    init_main()

def run_general():
    general_main()