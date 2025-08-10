"""
EnergyNexus Data Collection System - Diagnostic Test Script
===========================================================

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Institution: Queen Mary University of London
Program: MSc Data Science and AI - 2024/25

Purpose:
This diagnostic script validates the EnergyNexus data collection infrastructure
by testing API connectivity, module imports, configuration loading, and basic
data generation capabilities. It provides comprehensive error reporting and
troubleshooting guidance for system setup issues.

Validation Components:
1. Project directory structure verification
2. Python module import validation  
3. API configuration and credential loading
4. Data collector initialization testing
5. Synthetic data generation verification
6. File I/O and storage system testing

Usage:
    python test_download.py

Prerequisites:
    - Python 3.8+ with required dependencies installed
    - Valid API keys configured in config/api_keys.yaml
    - Proper project directory structure established

Exit Codes:
    0: All tests passed successfully
    1: Critical system failure detected
"""

import sys
import os
import traceback
from datetime import datetime

def print_header():
    """Print professional header for diagnostic output."""
    print("=" * 70)
    print("EnergyNexus Data Collection System - Diagnostic Test")
    print("=" * 70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 70)

def test_directory_structure():
    """
    Validate project directory structure and create missing directories.
    
    Returns:
        bool: True if directory structure is valid, False otherwise
    """
    print("\n[TEST 1] Project Directory Structure Validation")
    print("-" * 50)
    
    required_directories = [
        'data/raw',
        'data/processed', 
        'data/metadata',
        'logs',
        'results/reports',
        'config',
        'src',
        'src/data_pipeline',
        'src/data_pipeline/collectors'
    ]
    
    missing_directories = []
    created_directories = []
    
    for directory in required_directories:
        if not os.path.exists(directory):
            missing_directories.append(directory)
            try:
                os.makedirs(directory, exist_ok=True)
                created_directories.append(directory)
                print(f"  Created missing directory: {directory}")
            except Exception as e:
                print(f"  ERROR: Failed to create directory {directory}: {e}")
                return False
        else:
            print(f"  Verified existing directory: {directory}")
    
    if created_directories:
        print(f"  Successfully created {len(created_directories)} missing directories")
    
    print("  RESULT: Directory structure validation PASSED")
    return True

def test_project_structure():
    """
    Verify essential project files and source code structure.
    
    Returns:
        bool: True if project structure is valid, False otherwise
    """
    print("\n[TEST 2] Project File Structure Verification")
    print("-" * 50)
    
    essential_files = {
        'src/data_pipeline/collectors/eia_collector.py': 'EIA data collector module',
        'config/api_keys.yaml': 'API configuration file',
        'src/__init__.py': 'Source package initializer',
        'src/data_pipeline/__init__.py': 'Data pipeline package initializer',
        'src/data_pipeline/collectors/__init__.py': 'Collectors package initializer'
    }
    
    missing_files = []
    
    for file_path, description in essential_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  Verified: {file_path} ({file_size} bytes) - {description}")
        else:
            missing_files.append(file_path)
            print(f"  MISSING: {file_path} - {description}")
    
    if missing_files:
        print(f"  ERROR: {len(missing_files)} essential files are missing")
        print("  Missing files must be created before proceeding")
        return False
    
    print("  RESULT: Project structure verification PASSED")
    return True

def test_module_imports():
    """
    Test Python module imports and dependency availability.
    
    Returns:
        bool: True if all imports successful, False otherwise
    """
    print("\n[TEST 3] Python Module Import Validation")
    print("-" * 50)
    
    # Test standard library imports
    standard_modules = ['sys', 'os', 'json', 'datetime', 'logging']
    for module in standard_modules:
        try:
            __import__(module)
            print(f"  Standard library: {module} - AVAILABLE")
        except ImportError as e:
            print(f"  ERROR: Standard library {module} import failed: {e}")
            return False
    
    # Test required dependencies
    dependencies = [
        ('pandas', 'Data manipulation and analysis'),
        ('requests', 'HTTP client for API requests'),
        ('yaml', 'YAML configuration file parsing'),
        ('numpy', 'Numerical computing support')
    ]
    
    missing_dependencies = []
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"  Dependency: {package} - AVAILABLE ({description})")
        except ImportError:
            missing_dependencies.append(package)
            print(f"  MISSING: {package} - {description}")
    
    if missing_dependencies:
        print(f"  ERROR: {len(missing_dependencies)} required dependencies missing")
        print("  Install with: pip install " + " ".join(missing_dependencies))
        return False
    
    # Test project module imports
    print("  Testing project-specific imports...")
    sys.path.append('src')
    
    try:
        from data_pipeline.collectors.eia_collector import EIADataCollector
        print("  Project module: EIADataCollector - IMPORT SUCCESSFUL")
    except ImportError as e:
        print(f"  ERROR: EIADataCollector import failed: {e}")
        print("  Check that src/data_pipeline/collectors/eia_collector.py exists")
        return False
    except Exception as e:
        print(f"  ERROR: Unexpected error importing EIADataCollector: {e}")
        return False
    
    print("  RESULT: Module import validation PASSED")
    return True

def test_configuration_loading():
    """
    Test configuration file loading and API key validation.
    
    Returns:
        tuple: (success_status, config_data)
    """
    print("\n[TEST 4] Configuration Loading and Validation")
    print("-" * 50)
    
    config_path = 'config/api_keys.yaml'
    
    if not os.path.exists(config_path):
        print(f"  ERROR: Configuration file not found: {config_path}")
        print("  Creating sample configuration file...")
        
        sample_config = """# EnergyNexus API Configuration
eia:
  api_key: "A1KZLB3WdMAFgDNUrpkE8EW33knFLZCzTEQ5dywB"
  description: "U.S. Energy Information Administration API"
  
nrel:
  api_key: "d7RGasq4yNqlCabEFOCpFENrYqHLcqz3wcGtoydC"
  description: "National Renewable Energy Laboratory API"
"""
        try:
            with open(config_path, 'w') as f:
                f.write(sample_config)
            print(f"  Created configuration file: {config_path}")
        except Exception as e:
            print(f"  ERROR: Failed to create configuration file: {e}")
            return False, None
    
    # Load and validate configuration
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  Configuration file loaded successfully: {config_path}")
        
        # Validate EIA configuration
        if 'eia' in config and 'api_key' in config['eia']:
            eia_key = config['eia']['api_key']
            if eia_key and eia_key != "YOUR_EIA_API_KEY_HERE":
                print(f"  EIA API key: {eia_key[:10]}... (VALID FORMAT)")
            else:
                print("  WARNING: EIA API key appears to be placeholder")
        else:
            print("  WARNING: EIA configuration section missing")
        
        # Validate NREL configuration
        if 'nrel' in config and 'api_key' in config['nrel']:
            nrel_key = config['nrel']['api_key']
            if nrel_key and nrel_key != "YOUR_NREL_API_KEY_HERE":
                print(f"  NREL API key: {nrel_key[:10]}... (VALID FORMAT)")
            else:
                print("  WARNING: NREL API key appears to be placeholder")
        else:
            print("  WARNING: NREL configuration section missing")
        
        print("  RESULT: Configuration loading PASSED")
        return True, config
        
    except Exception as e:
        print(f"  ERROR: Configuration loading failed: {e}")
        return False, None

def test_collector_initialization():
    """
    Test EIA data collector initialization and basic functionality.
    
    Returns:
        tuple: (success_status, collector_instance)
    """
    print("\n[TEST 5] Data Collector Initialization")
    print("-" * 50)
    
    try:
        from data_pipeline.collectors.eia_collector import EIADataCollector
        
        # Initialize collector
        print("  Initializing EIADataCollector...")
        collector = EIADataCollector()
        print("  Data collector initialization: SUCCESSFUL")
        
        # Validate collector attributes
        if hasattr(collector, 'api_key'):
            if collector.api_key:
                print(f"  API key loaded in collector: {collector.api_key[:10]}...")
            else:
                print("  WARNING: No API key loaded in collector")
        
        if hasattr(collector, 'base_url'):
            print(f"  Base URL configured: {collector.base_url}")
        
        if hasattr(collector, 'session'):
            print("  HTTP session initialized: YES")
        
        print("  RESULT: Collector initialization PASSED")
        return True, collector
        
    except Exception as e:
        print(f"  ERROR: Collector initialization failed: {e}")
        print(f"  Exception details: {traceback.format_exc()}")
        return False, None

def test_data_generation_and_storage():
    """
    Test synthetic data generation and file storage capabilities.
    
    Args:
        collector: Initialized EIADataCollector instance
        
    Returns:
        bool: True if data operations successful, False otherwise
    """
    print("\n[TEST 6] Data Generation and Storage Validation")
    print("-" * 50)
    
    try:
        from data_pipeline.collectors.eia_collector import EIADataCollector
        collector = EIADataCollector()
        
        # Generate synthetic test data
        print("  Generating synthetic test dataset...")
        test_data = collector.create_synthetic_demand_data(
            hours=48,  # Two days of data
            base_demand=400,
            region='TEST_REGION'
        )
        
        if test_data is not None and len(test_data) > 0:
            record_count = len(test_data)
            min_demand = test_data['electricity_demand_mw'].min()
            max_demand = test_data['electricity_demand_mw'].max()
            
            print(f"  Synthetic data generated: {record_count} records")
            print(f"  Demand range: {min_demand:.1f} - {max_demand:.1f} MW")
            
            # Test data storage
            print("  Testing data storage functionality...")
            storage_result = collector.save_collected_data(test_data, "diagnostic_test_data")
            
            if storage_result:
                test_file_path = 'data/raw/diagnostic_test_data.csv'
                if os.path.exists(test_file_path):
                    file_size = os.path.getsize(test_file_path) / 1024  # KB
                    print(f"  Data file created: {test_file_path} ({file_size:.1f} KB)")
                    
                    # Cleanup test file
                    os.remove(test_file_path)
                    print("  Test file cleaned up successfully")
                    
                    print("  RESULT: Data generation and storage PASSED")
                    return True
                else:
                    print("  ERROR: Data file not found after save operation")
                    return False
            else:
                print("  ERROR: Data storage operation failed")
                return False
        else:
            print("  ERROR: Synthetic data generation failed")
            return False
            
    except Exception as e:
        print(f"  ERROR: Data generation test failed: {e}")
        print(f"  Exception details: {traceback.format_exc()}")
        return False

def generate_diagnostic_report(test_results):
    """
    Generate comprehensive diagnostic report and recommendations.
    
    Args:
        test_results (dict): Results from all diagnostic tests
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Tests Executed: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    test_names = {
        'directory_structure': 'Directory Structure Validation',
        'project_structure': 'Project File Structure Verification', 
        'module_imports': 'Python Module Import Validation',
        'configuration': 'Configuration Loading and Validation',
        'collector_init': 'Data Collector Initialization',
        'data_operations': 'Data Generation and Storage Validation'
    }
    
    for test_key, test_name in test_names.items():
        status = "PASSED" if test_results.get(test_key, False) else "FAILED"
        print(f"  {test_name}: {status}")
    
    if success_rate == 100:
        print("\nSYSTEM STATUS: All diagnostic tests passed successfully")
        print("RECOMMENDATION: Proceed with full data collection operations")
        print("NEXT STEPS:")
        print("  1. Execute: python scripts/download_eia_data.py")
        print("  2. Monitor logs for any operational issues")
        print("  3. Validate collected data quality")
    else:
        print("\nSYSTEM STATUS: One or more diagnostic tests failed")
        print("RECOMMENDATION: Address failed tests before proceeding")
        print("TROUBLESHOOTING STEPS:")
        
        if not test_results.get('module_imports', True):
            print("  1. Install missing dependencies: pip install requests pandas pyyaml")
        if not test_results.get('configuration', True):
            print("  2. Verify API keys in config/api_keys.yaml")
        if not test_results.get('project_structure', True):
            print("  3. Ensure all required source files are present")
        if not test_results.get('collector_init', True):
            print("  4. Check collector implementation for syntax errors")
            
    print("\nFor technical support, contact: Aditya Talekar (ec24018@qmul.ac.uk)")

def main():
    """
    Main diagnostic execution function with comprehensive error handling.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print_header()
    
    # Execute diagnostic test suite
    test_results = {}
    
    try:
        # Test 1: Directory structure
        test_results['directory_structure'] = test_directory_structure()
        
        # Test 2: Project structure  
        test_results['project_structure'] = test_project_structure()
        
        # Test 3: Module imports
        test_results['module_imports'] = test_module_imports()
        
        # Test 4: Configuration loading
        config_success, config_data = test_configuration_loading()
        test_results['configuration'] = config_success
        
        # Test 5: Collector initialization
        collector_success, collector = test_collector_initialization()
        test_results['collector_init'] = collector_success
        
        # Test 6: Data operations
        test_results['data_operations'] = test_data_generation_and_storage()
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Diagnostic execution failed: {e}")
        print(f"Exception details: {traceback.format_exc()}")
        return 1
    
    # Generate comprehensive report
    generate_diagnostic_report(test_results)
    
    # Determine exit code based on results
    all_passed = all(test_results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    """
    Script entry point with proper error handling and exit code management.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during diagnostic execution: {e}")
        sys.exit(1)