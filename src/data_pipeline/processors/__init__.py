"""
Data Pipeline Processors Package
EnergyNexus MSc Project

This package contains data processing modules for hybrid energy systems.
The processors handle specialized cleaning, transformation, and validation
of energy data from multiple sources.

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25
"""

from .data_cleaner import EnergyDataProcessor

# Package version for thesis documentation
__version__ = "1.0.0"

# Make main classes available at package level
__all__ = [
    "EnergyDataProcessor"
]

# Package metadata for academic documentation
__author__ = "Aditya Talekar"
__email__ = "ec24018@qmul.ac.uk"
__institution__ = "Queen Mary University of London"
__project__ = "EnergyNexus: Advanced Scheduling Algorithms for Integrated Power Systems"