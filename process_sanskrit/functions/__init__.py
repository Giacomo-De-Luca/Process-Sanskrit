"""Public processing functions, imported lazily to keep startup lightweight."""

from importlib import import_module

from process_sanskrit.functions.process import process


_EXPORTS = {
    "dict_search": ("process_sanskrit.functions.dictionaryLookup", "dict_search"),
    "multidict": ("process_sanskrit.functions.dictionaryLookup", "multidict"),
    "root_any_word": ("process_sanskrit.functions.rootAnyWord", "root_any_word"),
    "hybrid_sandhi_splitter": (
        "process_sanskrit.functions.hybridSplitter",
        "hybrid_sandhi_splitter",
    ),
    "inflect": ("process_sanskrit.functions.inflect", "inflect"),
    "clean_results": ("process_sanskrit.functions.cleanResults", "clean_results"),
    "root_compounds": (
        "process_sanskrit.functions.compoundAnalysis",
        "root_compounds",
    ),
    "process_root_result": (
        "process_sanskrit.functions.compoundAnalysis",
        "process_root_result",
    ),
}
__all__ = ["process", *_EXPORTS]


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
