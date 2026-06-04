from hindibabynet_cdi import plotting, scoring


def test_scoring_module_imports() -> None:
    assert scoring.__doc__ is not None
    assert plotting.__doc__ is not None