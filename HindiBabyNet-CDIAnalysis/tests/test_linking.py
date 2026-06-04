from hindibabynet_cdi import linking, metadata


def test_linking_module_imports() -> None:
    assert linking.__doc__ is not None
    assert metadata.__doc__ is not None