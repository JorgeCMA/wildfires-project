from pathlib import Path

import pytest
import yaml

from wildfire.config import load_config, PROJECT_ROOT, CONFIG_PATH


class TestProjectRoot:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_project_root_is_directory(self):
        assert PROJECT_ROOT.is_dir()

    def test_project_root_has_pyproject_toml(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()


class TestConfigPath:
    def test_config_path_exists(self):
        assert CONFIG_PATH.exists()

    def test_config_path_is_file(self):
        assert CONFIG_PATH.is_file()

    def test_config_path_is_yaml(self):
        assert CONFIG_PATH.suffix in (".yaml", ".yml")


class TestLoadConfig:
    def test_load_config_returns_dict(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_required_keys(self):
        config = load_config()
        assert "data" in config
        assert "firms" in config
        assert "clcplus" in config

    def test_load_config_data_paths(self):
        config = load_config()
        assert "raw" in config["data"]
        assert "intermediate" in config["data"]
        assert "processed" in config["data"]

    def test_load_config_clcplus_version(self):
        config = load_config()
        assert config["clcplus"]["version"] == 2023


class TestDirectoryStructure:
    @pytest.mark.parametrize("path", [
        "data/raw/firms",
        "data/raw/clcplus/2023/tile_01",
        "data/raw/clcplus/2023/tile_02",
        "data/intermediate/firms",
        "data/intermediate/clcplus",
        "data/processed/fires",
        "data/processed/datasets",
        "notebooks",
        "src/wildfire",
        "src/wildfire/data",
        "src/wildfire/geo",
        "src/wildfire/enrichment",
        "src/wildfire/processing",
        "src/wildfire/analysis",
        "scripts",
        "tests",
        "configs",
        "versioning",
    ])
    def test_directory_exists(self, path):
        full_path = PROJECT_ROOT / path
        assert full_path.exists(), f"Directory not found: {path}"
        assert full_path.is_dir(), f"Not a directory: {path}"

    @pytest.mark.parametrize("path", [
        "data/raw/firms/.gitkeep",
        "data/raw/clcplus/2023/tile_01/.gitkeep",
        "data/raw/clcplus/2023/tile_02/.gitkeep",
        "data/intermediate/firms/.gitkeep",
        "data/intermediate/clcplus/.gitkeep",
        "data/processed/fires/.gitkeep",
        "data/processed/datasets/.gitkeep",
    ])
    def test_gitkeep_exists(self, path):
        full_path = PROJECT_ROOT / path
        assert full_path.exists(), f".gitkeep not found: {path}"


class TestModuleImports:
    def test_import_wildfire(self):
        import wildfire
        assert wildfire is not None

    def test_import_config(self):
        from wildfire.config import load_config
        assert callable(load_config)

    def test_import_data(self):
        import wildfire.data
        assert wildfire.data is not None

    def test_import_geo(self):
        import wildfire.geo
        assert wildfire.geo is not None

    def test_import_enrichment(self):
        import wildfire.enrichment
        assert wildfire.enrichment is not None

    def test_import_processing(self):
        import wildfire.processing
        assert wildfire.processing is not None

    def test_import_analysis(self):
        import wildfire.analysis
        assert wildfire.analysis is not None


class TestVersioningCatalog:
    def test_catalog_exists(self):
        catalog_path = PROJECT_ROOT / "versioning" / "catalog.yaml"
        assert catalog_path.exists()

    def test_catalog_is_valid_yaml(self):
        catalog_path = PROJECT_ROOT / "versioning" / "catalog.yaml"
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)
        assert isinstance(catalog, dict)

    def test_catalog_has_datasets(self):
        catalog_path = PROJECT_ROOT / "versioning" / "catalog.yaml"
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)
        assert "datasets" in catalog
        assert "firms" in catalog["datasets"]
        assert "clcplus" in catalog["datasets"]
