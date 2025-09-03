import argparse
from pathlib import Path

import pytest

from sdk.helpers.config import ConfigLoader


class TestConfigLoader:
    def test_config_file_not_found(self, tmp_path: Path):
        non_existant_config_file = tmp_path / "config.toml"
        with pytest.raises(FileNotFoundError) as e:
            ConfigLoader(non_existant_config_file)
        assert f"Expected config file at {non_existant_config_file}" in str(e.value)

    def test_empty_config_file(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        config = ConfigLoader(config_file)
        assert config.model_path == "./models/network.rpk"
        assert not config.profiling
        assert config.width == 4056
        assert config.height == 3040

    def test_with_config_file(self, data_dir: Path):
        config_file = data_dir / "config/changed_default_config.toml"
        config = ConfigLoader(config_file)
        assert config.model_path == "different_model.rpk"
        assert config.profiling
        assert config.width == 1000
        assert config.height == 1000

    def test_with_args(self, tmp_path: Path):
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
        parser.add_argument("--width", type=int)
        parser.add_argument("--model_path", type=str)
        args = parser.parse_args(
            [
                "--debug",
                "--width",
                "1100",
                "--model_path",
                "different_model.rpk",
            ]
        )

        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        config = ConfigLoader(config_file, args)
        assert config.model_path == "different_model.rpk"
        assert not config.profiling
        assert config.width == 1100
        assert config.height == 3040

    def test_with_config_file_args(self, data_dir: Path):
        config_file = data_dir / "config/changed_default_config.toml"
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
        parser.add_argument("--width", type=int)
        parser.add_argument("--model_path", type=str)
        args = parser.parse_args(
            [
                "--no-debug",
                "--width",
                "1100",
                "--model_path",
                "wow.rpk",
            ]
        )
        config = ConfigLoader(config_file, args)
        assert config.model_path == "wow.rpk"
        assert config.profiling
        assert config.width == 1100
        assert config.height == 1000

    def test_extra_config_file_args(self, tmp_path: Path):
        parser = argparse.ArgumentParser()
        parser.add_argument("--wow1", type=str)
        args = parser.parse_args(
            [
                "--wow1",
                "new_arg",
            ]
        )
        config_file = tmp_path / "config.toml"
        config_file.write_text("wow2=3")
        config = ConfigLoader(config_file, args)
        assert config.config_data["wow2"] == 3
        assert config.config_data["wow1"] == "new_arg"

    def test_extend_config(self, tmp_path: Path):
        class ExtendConfigLoad(ConfigLoader):
            def __init__(self, config_file_path="./config.toml", args=None):
                super().__init__(config_file_path, args)
                self.some_path: Path = Path(self.config_data["some_path"])

            def default_config(self):
                return {**super().default_config(), "some_path": "./wow_path"}

        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        config = ExtendConfigLoad(config_file)
        assert config.some_path.as_posix() == Path("./wow_path").as_posix()

    def test_extend_config_file(self, tmp_path: Path):
        class ExtendConfigLoad(ConfigLoader):
            def __init__(self, config_file_path="./config.toml", args=None):
                super().__init__(config_file_path, args)
                self.some_path: Path = Path(self.config_data["some_path"])

            def default_config(self):
                return {**super().default_config(), "some_path": "./wow_path"}

        config_file = tmp_path / "config.toml"
        config_file.write_text('some_path = "../new_path"')
        config = ExtendConfigLoad(config_file)
        assert config.some_path.as_posix() == Path("../new_path").as_posix()
