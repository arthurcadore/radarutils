# Installation Guide

## Using pip (Recommended)

Install the latest version from PyPI:

```bash
pip install radarutils
```

## Installing from Source

For the latest development version:

```bash
git clone https://github.com/arthurcadore/radarutils.git
cd radarutils
pip install -e .
```

## Quick Start

### Basic Usage

```python
import radarutils
import numpy as np

# Calculate unambiguous range
pulse_period = 1e-3  # 1 ms
max_range = radarutils.core.basics.calc_unambiguous_range(pulse_period)
print(f"Maximum unambiguous range: {max_range/1000:.1f} km")

# Calculate antenna effective area
gain_db = 20  # 20 dB gain
frequency = 10e9  # 10 GHz
effective_area = radarutils.core.basics.effective_area(gain_db, frequency)
print(f"Effective antenna area: {effective_area:.2f} m²")
```

## Dependencies

RadarUtils requires:

- **Python 3.9+**
- **numpy** (≥1.21.0)
- **matplotlib** (≥3.4.0)
- **scipy** (≥1.7.0)
- **scienceplots** (≥2.1.1)

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you have Python 3.9+ installed
2. **Visualization Issues**: Install matplotlib backend for your system
3. **Performance**: Use numpy arrays for large-scale simulations

### Getting Help

- **Documentation**: [https://arthurcadore.github.io/radarutils/](https://arthurcadore.github.io/radarutils/)
- **Issues**: [GitHub Issues](https://github.com/arthurcadore/radarutils/issues)
- **Examples**: Check the `examples/` directory in the repository

## Next Steps

- Explore the [API Reference](../api/) for detailed function documentation
- Check out the [examples](../examples/) for complete simulation scripts
- Read the [changelog](changelog.md) for version history and updates