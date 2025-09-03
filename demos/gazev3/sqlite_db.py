"""
Controls the behaviour for how sqlite_db behaviour
"""
from .common import AdditionalDemoConfig
from pathlib import Path

class SDKArtifactDir:
    """
    Ensures the artifact folder from the sdk is created and stores the path
    Regardless of the config
    """
    def __init__(self, config: AdditionalDemoConfig):
        self.artifact_folder = Path(config.generate_artifacts.artifact_directory)
        self.artifact_folder.mkdir(parents=True, exist_ok=True)
