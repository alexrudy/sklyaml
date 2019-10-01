import functools
import importlib
from pathlib import Path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
import warnings
import copy
from collections import OrderedDict

import yaml

__all__ = ['load', 'Loader']

class NotAnEstimatorWarning(Warning):
    pass

def iter_candidates(include, *paths, missing=False):
    for path in paths:
        candidate = Path(path) / include
        if missing or candidate.exists():
            yield candidate


def find_include(include, *paths):
    """Find an include specified in a YAML file.
    
    Includes are searched first in `paths` provided to this function,
    then in the current working directory.
    """

    for candidate in iter_candidates(
        include, *paths, Path.cwd()
    ):
        return candidate
    raise FileNotFoundError(f"Can't find incldue {include} in {paths!r}")

_PRELOAD_TAGS = []
def register(loader):
    global _PRELOAD_TAGS
    for tag, constructor in _PRELOAD_TAGS:
        loader.add_constructor(tag, constructor)
    _PRELOAD_TAGS = []


class Loader(yaml.FullLoader):

    def tag(cls, tag=None):

        if tag is None:
            cls, tag = None, cls
        
        def inner(f):
            if cls is None:
                _PRELOAD_TAGS.append((tag, f))
            else:
                cls.add_constructor(tag, f)
            return f
        
        return inner
            

    @tag("!include")
    def skl_construct_include(self, node):
        path = find_include(node.value, Path(self.name).parent)
        with open(path, 'r') as stream:
            mapping = yaml.load(stream, Loader=self.__class__)
        return mapping

    @tag("!estimator")
    def skl_construct_class(self, node):
        properties = self.construct_mapping(node, deep=True)
        cls = resolve(properties.pop("()"))
        if not issubclass(cls, BaseEstimator):
            warnings.warn(NotAnEstimatorWarning("Class {!r} is not an estimator".format(cls)))

        return cls(**properties)

    @tag("!pipeline")
    def skl_construct_pipeline(self, node):
        mapping = self.construct_mapping(node, deep=True)
        steps = mapping.pop("steps")
        if isinstance(steps, dict):
            steps = list(steps.items())
        return Pipeline(steps, **mapping)

    @tag("!union")
    def skl_construct_union(self, node):
        mapping = self.construct_mapping(node, deep=True)
        steps = mapping.pop("steps", [])
        if isinstance(steps, dict):
            steps = list(steps.items())
        return FeatureUnion(steps, **mapping)

    @tag("!resolve")
    def skl_construct_class(self, node):
        return resolve(self.construct_scalar(node))

    @tag("!function")
    def skl_construct_function(self, node):
        kwargs = self.construct_mapping(node, deep=True)
        f = resolve(kwargs.pop("(f)"))
        args = kwargs.pop("(*)", ())
        if args or kwargs:
            return functools.partial(f, *args, **kwargs)
        return f

    tag = classmethod(tag)

register(Loader)

class Representer(yaml.SafeDumper):

    @classmethod
    def represents(dumper, cls, multi=False):
        def inner(f):
            if multi:
                dumper.add_multi_representer(cls, f)
            else:
                dumper.add_representer(cls, f)
            return f
        return inner

    @staticmethod
    def _qualname(obj):
        cls = type(obj)
        return "{}.{}".format(cls.__module__, cls.__qualname__)

    def represent_pipeline(self, data):
        properties = data.get_params(deep=False)
        properties['steps'] =  OrderedDict(properties['steps'])
        return self.represent_mapping("!pipeline", properties)
        
    def represent_estimator(self, data):
        properties = data.get_params(deep=False)
        assert '()' not in properties
        properties['()'] = self._qualname(data)
        return self.represent_mapping("!estimator", properties)

    def represent_union(self, data):
        properties = data.get_params(deep=False)
        properties['steps'] = dict(properties.pop("transformer_list", []))
        return self.represent_mapping("!union", properties)

    def represent_odict(self, data):
        return self.represent_mapping(self.DEFAULT_MAPPING_TAG, data.items())


Representer.add_multi_representer(BaseEstimator, Representer.represent_estimator)
Representer.add_representer(Pipeline, Representer.represent_pipeline)
Representer.add_representer(FeatureUnion, Representer.represent_union)
Representer.add_representer(OrderedDict, Representer.represent_odict)

load = functools.partial(yaml.load, Loader=Loader)
dump = functools.partial(yaml.dump, Dumper=Representer)

def resolve(name):
    """Resolve a qualified python name to an object."""
    if ":" in name:
        modnaem, clsname = name.rsplit(":", 1)
    else:
        modname, clsname = name.rsplit(".", 1)
    mod = importlib.import_module(modname)
    return getattr(mod, clsname)