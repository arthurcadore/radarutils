# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.10] - 2026-03-25

### News
- **Package Structure**: Complete reorganization into three main modules:
  - `core/`: Contains basics, data, and env_vars modules
  - `radar_components/`: Contains antenna, scene, target, wave, and radar modules
  - `visualization/`: Contains animations and plotter modules
- **Automated Testing**: Full pytest integration with:
  - Unit tests for all core functions
  - Import validation tests
  - Coverage reporting (~21% coverage)
  - GitHub Actions CI/CD pipeline
- **CI/CD Pipeline**: 
  - Multi-version Python testing (3.12)
  - Automatic package building
  - PyPI publishing on releases
  - Codecov integration
- **Package Metadata**: Enhanced `__init__.py` with proper exports and versioning

### Fixed
- **Circular Imports**: Resolved import cycles by moving imports inside functions
- **Module Organization**: Fixed module dependencies and cross-references
- **Test Compatibility**: Updated all tests to work with new package structure
- **Minimum Python**: Updated from 3.8 to 3.9 for better compatibility
- **Dependencies**: Added pytest and pytest-cov as development dependencies
- **Build System**: Enhanced pyproject.toml configuration
- **Import Structure**: Standardized to use `import radarutils` pattern
- **CI/CD Push flow to PyPi**: fixed legacy push.




## [1.0.7] - 2026-03-01
- Initial radar simulation functionality
- Basic antenna patterns and wave propagation
- Scene and target modeling
- Plotting and animation tools
