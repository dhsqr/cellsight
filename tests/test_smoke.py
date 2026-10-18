import pytest
import cellsight

def test_add():
    assert cellsight.add(2, 3) == 5

def test_add_rejects_string():
    with pytest.raises(TypeError):
        cellsight.add("a", 3)