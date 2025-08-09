# plugin_loader.py
import importlib
import os
from typing import Any, Dict

PLUGINS_PACKAGE = "plugins"

def load_plugin(plugin_path: str, **kwargs) -> Any:
    """
    Load plugin by 'module.ClassName' from plugins/ package.
    """
    if not plugin_path or '.' not in plugin_path:
        raise ValueError("plugin_path must be 'module.ClassName'")

    module_name, class_name = plugin_path.rsplit('.', 1)
    full_module = f"{PLUGINS_PACKAGE}.{module_name}"

    if full_module.split('.')[0] != PLUGINS_PACKAGE:
        raise ValueError(f"Plugins must be in {PLUGINS_PACKAGE} package")

    module = importlib.import_module(full_module)
    plugin_cls = getattr(module, class_name)
    return plugin_cls(**kwargs)

def list_available_plugins() -> Dict[str, str]:
    pkg_dir = os.path.join(os.path.dirname(__file__), PLUGINS_PACKAGE)
    plugins = {}
    if os.path.isdir(pkg_dir):
        for fn in os.listdir(pkg_dir):
            if fn.endswith('.py') and fn != "__init__.py":
                name = fn[:-3]
                plugins[name] = f"{PLUGINS_PACKAGE}.{name}"
    return plugins
