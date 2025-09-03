import importlib
import importlib.util
import sys
from pathlib import Path


def import_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path.absolute().as_posix()
    )
    module = importlib.util.module_from_spec(spec)
    # FIX: Bug when multiple demo_pipelines are loaded, we need to manually clear the old reference and other shared module namespace
    demo_pipeline_modules = list(
        filter(lambda module: "demo_pipeline" in module, sys.modules.keys())
    )
    for demo_pipeline_module in demo_pipeline_modules:
        del sys.modules[demo_pipeline_module]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
