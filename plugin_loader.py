# plugin_loader.py
import importlib
import os
from typing import Any, Dict

PLUGINS_PACKAGE = "plugins"

def load_plugin(plugin_type: str, plugin_name: str, base_class=None, class_name=None, **kwargs) -> Any:
    """
    Load a plugin by type and name from plugins/<plugin_type>/<plugin_name>.py
    """
    if plugin_type not in {"mutators", "generators", "detectors", "loggers"}:
        raise ValueError(f"Unknown plugin type: {plugin_type}")

    full_module = f"{PLUGINS_PACKAGE}.{plugin_type}.{plugin_name}"
    module = importlib.import_module(full_module)

    # Use provided class_name or generate from module name
    if class_name is None:
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

def load_detector(module_name: str, class_name: str = None, **kwargs):
    from plugin_base import DetectorPlugin
    return load_plugin("detectors", module_name, base_class=DetectorPlugin, class_name=class_name, **kwargs)

def load_mutator(module_name: str, class_name: str = None, **kwargs):
    from plugin_base import MutatorPlugin
    return load_plugin("mutators", module_name, base_class=MutatorPlugin, class_name=class_name, **kwargs)

def load_logger(module_name: str, class_name: str = None, **kwargs):
    from plugin_base import PluginBase
    return load_plugin("loggers", module_name, base_class=PluginBase, class_name=class_name, **kwargs)

def load_generator(module_name: str, class_name: str = None, **kwargs):
    from plugin_base import GeneratorBase
    return load_plugin("generators", module_name, base_class=GeneratorBase, class_name=class_name, **kwargs)