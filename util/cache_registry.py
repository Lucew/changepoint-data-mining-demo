import threading
import weakref
import functools
import typing as tp
import dataclasses
import logging

# We keep weak references so this registry does not keep functions alive.
# Each item is the cached function object returned by functools.lru_cache(...)
_Registry = weakref.WeakSet[tp.Callable[..., tp.Any]]
_registry: _Registry = weakref.WeakSet()
_lock = threading.RLock()
logger = logging.getLogger("frontend-logger")


def _register_cached_function(func: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
    """Add a cached function (i.e., the lru_cache-wrapped callable) to the registry."""
    with _lock:
        _registry.add(func)
    return func


def lru_cache(maxsize: tp.Optional[int] = 128, typed: bool = False) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]:
    """
    Decorator factory mirroring functools.lru_cache, but also registers the cached function
    so it can be cleared globally via clear_all_caches().

    Usage:
        @cached()                     # default maxsize=128
        @cached(maxsize=None)         # unbounded (like functools.cache)
        @cached(maxsize=1024, typed=True)
    """
    def decorator(fn: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
        # Apply lru_cache first so we return the real cached function that has .cache_clear/.cache_info
        cached_fn = functools.lru_cache(maxsize=maxsize, typed=typed)(fn)
        return _register_cached_function(cached_fn)
    return decorator


def cache(fn: tp.Optional[tp.Callable[..., tp.Any]] = None) -> tp.Callable[..., tp.Any]:
    """
    Drop-in equivalent of functools.cache (i.e., lru_cache(maxsize=None)) that also registers.
    Can be used with or without parentheses:

        @cache
        def f(...): ...

        @cache()
        def g(...): ...
    """
    def _apply(f: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
        cached_fn = functools.lru_cache(maxsize=None)(f)
        return _register_cached_function(cached_fn)

    if fn is None:
        return _apply
    return _apply(fn)


def clear_all_caches() -> None:
    """Clear every registered cache. Safe to call multiple times."""
    logger.info(f"[{__name__}] Clearing all caches.")
    with _lock:
        # Copy to a list so we don't mutate during iteration if something GC's
        for func in list(_registry):
            # lru_cache-wrapped callables have cache_clear()
            clear = getattr(func, "cache_clear", None)
            if callable(clear):
                clear()


def iter_cached_functions() -> tp.Iterable[tp.Callable[..., tp.Any]]:
    """Yield currently-registered cached functions (weakly held)."""
    with _lock:
        yield from list(_registry)


@dataclasses.dataclass
class CacheStats:
    hits: int
    misses: int
    maxsize: int
    currsize: int = None


def cache_stats() -> tp.Dict[str, CacheStats]:
    """
    Return {qualname: cache_info_tuple} for each registered function.
    Each tuple is (hits, misses, maxsize, currsize), matching functools._CacheInfo.
    """
    stats: tp.Dict[str, CacheStats] = {}
    with _lock:
        for func in list(_registry):
            info = getattr(func, "cache_info", None)
            if callable(info):
                ci = info()  # _CacheInfo(hits, misses, maxsize, currsize)
                # Try to get a helpful name
                name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
                # Include module if available to reduce ambiguity across files
                mod = getattr(func, "__module__", None)
                key = f"{mod}.{name}" if mod else name
                stats[key] = CacheStats(ci.hits, ci.misses, ci.maxsize, ci.currsize)
    return stats
