"""Shim: DNPG trainer/FSEval.py:21 imports pandas but never uses it.

Dummy classes only, so isinstance checks done by other libraries (e.g. sklearn's _is_pandas_df, which looks at
sys.modules['pandas'].DataFrame) are simply False."""
__version__ = "0.0-shim"


class DataFrame:  # noqa: D101
    pass


class Series:  # noqa: D101
    pass


class Index:  # noqa: D101
    pass
