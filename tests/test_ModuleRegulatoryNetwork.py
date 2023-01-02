import numpy as np
import pytest


@pytest.fixture
def test_import_network_edges_from_lpan(my_grn):
    my_grn.clean_up_network()
    # my_grn.plot_network()
    my_module_network = my_grn.get_module_module_network()
    # my_module_network.plot_network()
    return my_module_network

def test_convert_to_ode(test_import_network_edges_from_lpan):
    my_ode = test_import_network_edges_from_lpan.convert_to_ode()
    params = np.random.rand(my_ode.nr_params).tolist()
    print(my_ode(None, [0,1,0,0], params))

