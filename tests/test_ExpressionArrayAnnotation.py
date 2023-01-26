def test_probe_to_agi(my_expression_annotation):
    probe_id = '244919_at'
    assert 'ATMG00960' in my_expression_annotation.probe_to_agi(probe_id)
