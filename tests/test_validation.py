from src.validation import purged_walk_forward

def test_no_leakage():
    n, h = 10_000, 30
    for tr, te in purged_walk_forward(n, 5, h):
        assert len(set(tr) & set(te)) == 0
        assert tr.max() + h < te.min()   # last train label ends before test starts
