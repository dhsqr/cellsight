if __name__ == "__main__":
    from cellsight._core import add

    print(add(2, 3))
else:
    from ._core import add

__all__ = ["add"]
