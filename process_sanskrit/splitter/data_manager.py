"""Locate the splitter's data files inside the installed package."""

import importlib.resources


def data_file_path(filename: str) -> str:
    return str(importlib.resources.files(__package__).joinpath("data", filename))
