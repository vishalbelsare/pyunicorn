# This file is part of pyunicorn.
# Copyright (C) 2008--2025 Jonathan F. Donges and pyunicorn authors
# URL: <https://www.pik-potsdam.de/members/donges/software-2/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
This module provides the mix-in class `Cached`, which manages LRU caches for
derived quantity methods, with declared dependencies on mutable instance
attributes.
"""

from abc import ABC, abstractmethod
from functools import lru_cache, _lru_cache_wrapper, wraps
from weakref import ReferenceType, ref
from inspect import getmembers, ismethod
from typing import Any, Tuple, Optional
from collections.abc import Hashable


class CacheRef:
    """
    Internal coupling layer between the global (class level) and local
    (instance level) LRU caches maintained by a `Cached` subclass, i.e., a
    wrapper that weakly references a local cache and that is held as a value in
    a global cache.

    The purpose of this wrapper is:
      - to avoid strong references from a global to its local caches, so that
        the latter can be destroyed together with the `Cached` instances owning
        them,
      - and to delete local caches when the global cache is cleared.

    Implementation details
    ----------------------

    Rationale:

      Global cache:
        - logically associated to a `Cached` subclass method
        - owned (strongly referenced) by a `Cached` subclass method
        - populated by `@lru_cache(cached_global)`
        - holds as keys: weak references to `Cached` subclass instances
        - holds as values: `CacheRef` instances

      Local cache:
        - logically associated to a `Cached` subclass instance and method
        - owned (strongly referenced) by a `Cached` subclass instance
        - populated by `@lru_cache(cached_local)`
        - holds as keys: method cache keys, as defined by `cached_local()` args
        - holds as values: cached method results

    Network of strong (==>) and weak (-->) references:

      {obj}                          : <Cached object>
       ==> .__class__                : <Cached class>
        ==> .{method}                : <Cached.{method} method>
         ==> .<locals>.cached_global : _lru_cache_wrapper
          ==> .<locals>.cache        : dict
           ==> .keys()               : dict_keys[<weakref to Cached object>]
            --> ()                   : <Cached object>
           ==> .values()             : dict_values[<CacheRef object>]
            ==> .obj                 : <weakref to Cached object>
             --> ()                  : <Cached object>
       ==> .__cached_{method}__      : _lru_cache_wrapper
        ==> .<locals>.cache          : dict
         ==> .keys()                 : dict_keys[Tuple[...]]
         ==> .values()               : dict_values[Any]

    Call stack (~>) and runtime trigger (#>) during global clearing:

         <Cached.{method} method>.cache_clear()
      ~> <Cached.{method} method>.<locals>.cached_global.<locals>.cache.clear()
      #> <CacheRef object>.__del__()  # no references left
      ~> delattr(<Cached object>, "__cached_{method}__")
    """

    __slots__ = ["obj", "attr"]

    def __init__(self, obj, attr: str):
        assert isinstance(getattr(obj, attr), _lru_cache_wrapper)
        self.obj: ReferenceType = ref(obj)
        self.attr = attr

    @property
    def cache(self) -> _lru_cache_wrapper:
        return getattr(self.obj(), self.attr)

    def __del__(self):
        obj = self.obj()
        if obj is not None:
            delattr(obj, self.attr)


class Cached(ABC):
    """
    A mix-in class which manages, for each subclass method decorated with
    `@Cached.method()`, one global (class level) and multiple local (instance
    level) LRU caches that are implemented via `@functools.lru_cache()`. The
    global cache essentially implements a single dispatch mechanism.

    The caches are populated simply by calling the decorated instance methods
    with new arguments, and cache entries are invalidated when designated
    instance attributes are mutated.

    `Cached` guarantees a bounded number of global slots (cached instance) per
    method, and a bounded number of local slots (cached argument tuples) per
    method and instance. A local cache is destroyed together with its instance.
    Given an instance `x`, a method cache can also be cleared by calling
    `x.{method}.cache_clear()`, which destroys all associated local caches. All
    method caches can be cleared by calling `x.cache_clear()`. For
    implementation details, see `CacheRef`.

    To inherit these capabilities, a subclass needs to:

      - decorate derived quantity methods with `@Cached.method()`,
      - provide a method `Cached.__cache_state__() -> Tuple[Hashable,...]`,
        which is used by `Cached` to define the `__eq__()` and `__hash__()`
        methods required by `@functools.lru_cache()`.

    These mix-in class attributes affect subsequently decorated methods:

      - cache_enable:      toggles method caching
      - lru_params_global: sets global `@functools.lru_cache()` parameters
      - lru_params_local:  sets local `@functools.lru_cache()` parameters

    NOTE:

        The intended caching behaviour, including invalidation semantics, is
        specified by `tests/test_core/test_cache.py`.
    """

    cache_enable = True

    lru_params_global = {"maxsize": 16, "typed": False}
    lru_params_local = {"maxsize": 3, "typed": True}

    @abstractmethod
    def __cache_state__(self) -> Tuple[Hashable, ...]:
        """
        Hashable tuple of mutable object attributes, which will determine the
        instance identity for ALL cached method lookups in this class,
        *in addition* to the built-in object `id()`. Returning an empty tuple
        amounts to declaring the object immutable in general. Mutable
        dependencies that are specific to a method should instead be declared
        via `@Cached.method(attrs=(...))`.

        NOTE:

            A subclass is responsible for the consistency and cost of this
            state descriptor. For example, hashing a large array attribute may
            be circumvented by declaring it as a property, with a custom setter
            method that increments a dedicated mutation counter.
        """

    def __eq__(self, other):
        return (self is other) and (
            self.__cache_state__() == other.__cache_state__())

    def __hash__(self):
        return hash((id(self),) + self.__cache_state__())

    @classmethod
    def method(cls, name: Optional[str] = None,
               attrs: Optional[Tuple[str, ...]] = None):
        """
        Caching decorator based on `@functools.lru_cache()`.

        Cache entries for decorated methods are indexed by the combination of:

          - the object `id()`,
          - the object-level mutable instance attributes, as declared by the
            subclass method `__cache_state__()`,
          - the method-level mutable instance attributes, as declared by the
            optional decorator argument `attrs`, and
          - the argument pattern at the call site, including the ordering of
            named arguments.

        The decorated method provides several attributes of its own, as defined
        by `@functools.lru_cache()`, including:

          - `cache_clear()`: empty the method cache for ALL class instances
          - `__wrapped__`: undecorated original method

        :arg name: Optionally print a message at the first method invocation.
        :arg attrs: Optionally declare attribute names as mutable dependencies.

        NOTE:

            The same reasoning about consistency and cost applies to the
            `attrs` argument as to the `__cache_state__()` method.

        Implementation detail
        ---------------------
        The local cache can be accssed at `obj.__cached_{method_name}__`.
        """
        # Evaluated at decorator instantiation.
        attrs = () if attrs is None else attrs
        assert isinstance(attrs, tuple)
        assert all(isinstance(a, str) for a in attrs)
        assert name is None or isinstance(name, str)

        def wrapper(f):
            """ Evaluated at decorator application (method definition). """

            def uncached(self, name, *args, **kwargs) -> Any:
                """ Evaluated at uncached method invocation. """
                if name is not None and getattr(self, "silence_level", 0) <= 1:
                    print(f"Calculating {name}...")
                return f(self, *args, **kwargs)

            if cls.cache_enable:
                def create_local_cache(_self) -> _lru_cache_wrapper:
                    """ Evaluated at first global cache miss. """
                    # closure holding an instance weakref
                    @lru_cache(**cls.lru_params_local)
                    def cached_local(_h, name, _l, *args, **kwargs):
                        """ Evaluated at every local cache miss. """
                        # dereference instance, ignore hash after cache lookup,
                        # remove `attrs` from args
                        return uncached(_self(), name, *args[_l:], **kwargs)
                    return cached_local

                @lru_cache(**cls.lru_params_global)
                def cached_global(_self: ReferenceType) -> CacheRef:
                    """ Evaluated at every global cache miss. """
                    # dereference instance
                    self = _self()
                    assert self is not None, "encountered dead weakref"
                    # access or create local cache
                    cache_attr = f"__cached_{f.__name__}__"
                    cached_local = getattr(self, cache_attr, None)
                    if cached_local is None:
                        cached_local = create_local_cache(_self)
                        # store local cache as instance attribute
                        setattr(self, cache_attr, cached_local)
                    # store reference to local cache as global cache result
                    return CacheRef(self, cache_attr)

                @wraps(cached_global, assigned=('cache_info', 'cache_clear'))
                def wrapped(self, *args, **kwargs):
                    """ Evaluated at every decorated method invocation. """
                    # pass instance weakref, obtain local cache
                    cached_local = cached_global(ref(self)).cache
                    # pass instance hash, prepend `attrs` to args
                    return cached_local(
                        hash(self.__cache_state__()), name,
                        len(attrs), *(getattr(self, a) for a in attrs),
                        *args, **kwargs)

            else:
                def wrapped(self, *args, **kwargs):
                    return uncached(self, name, *args, **kwargs)

            # fully decorated method
            return wraps(f)(wrapped)
        return wrapper

    @staticmethod
    def is_global_cache(attr) -> bool:
        return ismethod(attr) and all(
            hasattr(attr, p) for p in ("cache_clear", "__wrapped__"))

    def cache_clear(self, prefix: Optional[str] = None) -> None:
        """
        Delete all method caches for ALL instances of `self.__class__`, and
        also recursively for any owned `Cached` instances listed in
        `self.__cache_state__()`. This is simply a loop over the
        `cache_clear()` methods for the individual cached methods.

        When a SINGLE instance is destroyed, its local method caches are
        deleted as well. Hence, this function is only useful in cases where the
        instances should be kept alive.

        :arg prefix: Optionally restrict the deleted caches by method name.

        NOTE:

            Instead, *invalidating* individual cache entries for a SINGLE
            instance is achieved by modifying the declared mutable attributes
            of that instance, see `@Cached.method()`.
        """
        for n, m in getmembers(self, predicate=self.is_global_cache):
            if prefix is None or n.startswith(prefix):
                m.cache_clear()
        for attr in self.__cache_state__():
            if isinstance(attr, Cached):
                attr.cache_clear(prefix=prefix)
