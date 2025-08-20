# plugin_loader.py
import importlib
import os
from typing import Any, Dict

PLUGINS_PACKAGE = "plugins"

def load_plugin(plugin_type: str, plugin_name: str, base_class=None, **kwargs) -> Any:
    """
    Load a plugin by type and name from plugins/<plugin_type>/<plugin_name>.py
    """
    if plugin_type not in {"mutators", "generators", "detector", "loggers"}:
        raise ValueError(f"Unknown plugin type: {plugin_type}")

    full_module = f"{PLUGINS_PACKAGE}.{plugin_type}.{plugin_name}"
    module = importlib.import_module(full_module)

    # assume class name is CamelCase version of module
    class_name = "".join(part.capitalize() for part in plugin_name.split("_"))
    plugin_cls = getattr(module, class_name)

    plugin = plugin_cls(**kwargs)

    if base_class and not isinstance(plugin, base_class):
        raise TypeError(f"{plugin_name} is not a {base_class.__name__}")

    return plugin

def list_available_plugins() -> Dict[str, str]:
    pkg_dir = os.path.join(os.path.dirname(__file__), PLUGINS_PACKAGE)
    plugins = {}
    if os.path.isdir(pkg_dir):
        for fn in os.listdir(pkg_dir):
            if fn.endswith('.py') and fn != "__init__.py":
                name = fn[:-3]
                plugins[name] = f"{PLUGINS_PACKAGE}.{name}"
    return plugins

def load_mutator(name: str, **kwargs):
    from plugin_base import MutatorPlugin
    return load_plugin("mutators", name, base_class=MutatorPlugin, **kwargs)

def load_detector(name: str, **kwargs):
    from plugin_base import DetectorPlugin
    return load_plugin("detector", name, base_class=DetectorPlugin, **kwargs)

def load_generator(name: str, **kwargs):
    from plugin_base import GeneratorBase
    return load_plugin("generators", name, base_class=GeneratorBase, **kwargs)

def load_logger(name: str, **kwargs):
    from plugin_base import BasePlugin
    return load_plugin("loggers", name, base_class=BasePlugin, **kwargs)