from hindibabynet_cdi import cleaning, io


def test_cleaning_module_imports() -> None:
    assert cleaning.__doc__ is not None
    assert io.__doc__ is not None