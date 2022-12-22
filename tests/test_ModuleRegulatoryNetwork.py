def test_import_network_edges_from_lpan(my_grn):
    my_grn.clean_up_network()
    my_grn.plot_network()
    assert len(my_grn.graph.nodes) > 0
    assert len(my_grn.graph.edges) > 0
