"""
Configuration management utilities.
Load and manage YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage YAML configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific config file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config: {config_name}")
        return config

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory.

        Returns:
            Dictionary mapping config names to their contents
        """
        configs = {}
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            configs[config_name] = self.load(config_name)

        logger.info(f"Loaded {len(configs)} configuration files")
        return configs

    def save(self, config: Dict[str, Any], config_name: str):
        """
        Save a configuration dictionary to a YAML file.

        Args:
            config: Configuration dictionary
            config_name: Name of config file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config: {config_name}")


def get_config(config_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load a single config file.

    Args:
        config_name: Name of config file
        config_dir: Directory containing config files

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)
