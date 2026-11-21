from steering.extract_features import Backend


def test_native_backend_value():
    assert Backend("native") is Backend.native
    assert Backend.native.value == "native"
    # existing members unchanged
    assert Backend("auto") is Backend.auto
    assert Backend("nnterp") is Backend.nnterp
    assert Backend("transformerlens") is Backend.transformerlens
