import pytest
import cellsight

def test_add():
    assert cellsight.add(2.5, 3.0) == 5.5

def test_add_rejects_string():
    with pytest.raises(TypeError):
        cellsight.add("a", 3)

def test_add_doubles():
    assert cellsight.add(2.5, 1.0) == 3.5
