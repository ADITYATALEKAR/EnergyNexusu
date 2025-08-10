"""
EnergyNexus Setup Configuration
===============================

Professional package setup for the EnergyNexus energy forecasting and optimization system.
This setup.py file configures the project for installation, dependency management, and distribution.

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Institution: Queen Mary University of London
Program: MSc Data Science and AI - 2024/25
Supervisor: Saqib Iqbal
Date: 2025
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Ensure we're using the correct directory
here = Path(__file__).parent.resolve()

# Get version information from the main package
sys.path.insert(0, str(here / 'src'))
try:
    from src import __version__, __author__, __email__, __title__, __description__
    version = __version__
    author = __author__
    author_email = __email__
    title = __title__
    description = __description__
except ImportError:
    # Fallback version information if import fails
    version = "1.0.0"
    author = "Aditya Talekar"
    author_email = "ec24018@qmul.ac.uk"
    title = "EnergyNexus"
    description = "Energy System Forecasting and Optimization Platform"

# Read the long description from README
long_description_file = here / "README.md"
if long_description_file.exists():
    with open(long_description_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = description

# Read requirements from requirements.txt if it exists
requirements_file = here / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    # Default requirements based on your project needs
    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.8.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
        "statsmodels>=0.13.0",
        "plotly>=5.10.0",
        "jupyterlab>=3.4.0"
    ]

# Development and testing requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "ipywidgets>=7.7.0"
]

# Documentation requirements
docs_requirements = [
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinxcontrib-napoleon>=0.7",
    "myst-parser>=0.17.0"
]

# Energy-specific optional requirements
energy_requirements = [
    "pvlib>=0.9.0",  # Solar energy modeling
    "windpowerlib>=0.2.0",  # Wind energy modeling
    "pyomo>=6.4.0",  # Optimization modeling
    "gurobipy>=9.5.0",  # Commercial optimization solver (optional)
]

# All optional requirements combined
all_requirements = dev_requirements + docs_requirements + energy_requirements

# Package data to include
package_data = {
    "": [
        "*.yml", "*.yaml", "*.json", "*.csv", "*.txt",
        "*.md", "*.rst", "*.cfg", "*.ini"
    ],
}

# Entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "energynexus-download=scripts.download_eia_data:main",
        "energynexus-quality=src.data_pipeline.processors.data_quality:main",
        "energynexus-train=src.forecasting.models.lstm_model:main",
        "energynexus-evaluate=src.evaluation.validators.model_evaluator:main",
    ],
}

# Classifiers for PyPI (if you decide to publish)
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

# Keywords for searchability
keywords = [
    "energy", "forecasting", "LSTM", "time-series", "renewable-energy",
    "electricity", "demand-prediction", "machine-learning", "optimization",
    "grid-analysis", "data-science", "neural-networks", "MSc-project"
]

# Project URLs
project_urls = {
    "Documentation": "https://github.com/aditya-talekar/EnergyNexus/docs",
    "Source": "https://github.com/aditya-talekar/EnergyNexus",
    "Tracker": "https://github.com/aditya-talekar/EnergyNexus/issues",
    "Research": "https://qmul.ac.uk/msc-data-science-ai",
}

# Validate Python version
if sys.version_info < (3, 8):
    raise RuntimeError("EnergyNexus requires Python 3.8 or later")

# Main setup configuration
setup(
    # Basic package information
    name=title.lower().replace(" ", "-"),
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    
    # Package description
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs and metadata
    url="https://github.com/aditya-talekar/EnergyNexus",
    project_urls=project_urls,
    
    # Package discovery and structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=package_data,
    include_package_data=True,
    
    # Requirements and dependencies
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "energy": energy_requirements,
        "all": all_requirements,
    },
    
    # Entry points and scripts
    entry_points=entry_points,
    scripts=[
        "scripts/download_eia_data.py",
    ],
    
    # Classification and keywords
    classifiers=classifiers,
    keywords=keywords,
    
    # License
    license="MIT",
    
    # Additional metadata
    platforms=["any"],
    zip_safe=False,
    
    # Test configuration
    test_suite="tests",
    tests_require=dev_requirements,
    
    # Academic and research metadata
    headers={
        "Academic-Project": "MSc Data Science and AI",
        "Institution": "Queen Mary University of London",
        "Supervisor": "Saqib Iqbal",
        "Year": "2024/25",
        "Research-Area": "Energy Forecasting and Optimization"
    }
)

# Post-installation message
def print_post_install_message():
    """Print helpful information after installation."""
    print("\n" + "="*60)
    print("EnergyNexus Installation Complete!")
    print("="*60)
    print("Energy System Forecasting and Optimization Platform")
    print(f"Version: {version}")
    print(f"Author: {author}")
    print("Institution: Queen Mary University of London")
    print("\nNext Steps:")
    print("1. Configure your EIA API key in config/api_keys.yaml")
    print("2. Run: energynexus-download to collect energy data")
    print("3. Open notebooks/01_data_exploration/ for analysis")
    print("4. Train models with: energynexus-train")
    print("\nDocumentation and examples available in notebooks/")
    print("="*60)

# Development setup utilities
class DevelopmentSetup:
    """Utilities for development environment setup."""
    
    @staticmethod
    def create_directory_structure():
        """Create the complete project directory structure."""
        directories = [
            "data/raw", "data/processed", "data/external", "data/synthetic",
            "models/lstm_models", "models/ensemble_models",
            "results/plots", "results/reports", "results/experiments",
            "logs/training", "logs/evaluation",
            "config", "tests/unit", "tests/integration",
            "docs/source", "docs/build"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("✓ Project directory structure created")
    
    @staticmethod
    def create_sample_configs():
        """Create sample configuration files."""
        config_files = {
            "config/api_keys.yaml": """
# EnergyNexus API Configuration
eia:
  api_key: "YOUR_EIA_API_KEY_HERE"
  base_url: "https://api.eia.gov/v2"
  note: "Get your free API key from https://www.eia.gov/opendata/register.php"

openmeteo:
  base_url: "https://archive-api.open-meteo.com/v1"
  note: "Open-Meteo provides free weather data"
""",
            "config/model_params.yaml": """
# EnergyNexus Model Configuration
lstm:
  sequence_length: 48
  forecast_horizons: [1, 6, 24]
  lstm_units: [64, 32]
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  target_variable: "energy_demand"
  feature_variables: ["energy_demand", "solar_generation", "wind_generation", "temperature"]
  quality_threshold: 0.8
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
""",
            "config/system_config.yaml": """
# EnergyNexus System Configuration
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  models: "models"
  results: "results"
  logs: "logs"

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  
monitoring:
  enable_tensorboard: true
  save_checkpoints: true
  evaluation_frequency: 10
"""
        }
        
        for filepath, content in config_files.items():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    f.write(content.strip())
        
        print("✓ Sample configuration files created")
    
    @staticmethod
    def setup_development_environment():
        """Complete development environment setup."""
        print("Setting up EnergyNexus development environment...")
        DevelopmentSetup.create_directory_structure()
        DevelopmentSetup.create_sample_configs()
        print("✓ Development environment setup complete!")

# Custom commands for setup.py
try:
    from setuptools import Command
    
    class DevelopmentSetupCommand(Command):
        """Custom command to set up development environment."""
        description = "Set up complete development environment"
        user_options = []
        
        def initialize_options(self):
            pass
        
        def finalize_options(self):
            pass
        
        def run(self):
            DevelopmentSetup.setup_development_environment()
    
    # Add custom command to setup
    setup.cmdclass = {'develop_setup': DevelopmentSetupCommand}
    
except ImportError:
    # setuptools.Command not available
    pass

# If run directly, provide information
if __name__ == "__main__":
    print("EnergyNexus Setup Configuration")
    print("=" * 40)
    print(f"Package: {title}")
    print(f"Version: {version}")
    print(f"Author: {author}")
    print(f"Python: {sys.version}")
    print("\nTo install the package:")
    print("  pip install -e .")
    print("\nTo install with development dependencies:")
    print("  pip install -e .[dev]")
    print("\nTo install with all optional dependencies:")
    print("  pip install -e .[all]")
    print("\nTo set up development environment:")
    print("  python setup.py develop_setup")