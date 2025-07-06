"""
Pytest configuration and shared fixtures for PII deidentification tests.
"""

import pytest
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List
import sys
import tempfile

# Add src to path for imports in a more secure way
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_transcript(test_data_dir):
    """Load sample transcript for testing."""
    with open(test_data_dir / "simple_transcript.txt", 'r') as f:
        return f.read()


@pytest.fixture(scope="session")
def expected_results(test_data_dir):
    """Load expected test results."""
    with open(test_data_dir / "expected_results.json", 'r') as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_ground_truth():
    """Sample ground truth data for testing - matches expected_results.json."""
    return {
        'member_first_name': 'Ella',
        'member_full_name': 'Ella Michael Wilson',
        'member_email': 'ella.wilson@example.com',
        'member_phone': '041648 996 374',
        'member_mobile': '041648 996 374',
        'member_address': '34 Church Street, Adelaide SA 5000',
        'member_number': '95924617',
        'agent_first_name': 'Ava',
        'consultant_first_name': 'Ava',
        'case_number': 'CASE-2024-001'
    }


@pytest.fixture(scope="session")
def sample_pii_detections():
    """Sample PII detections for testing."""
    return [
        {
            'text': 'Ella',
            'entity_type': 'PERSON',
            'start': 427,
            'end': 431,
            'score': 0.95
        },
        {
            'text': 'ella.wilson@example.com',
            'entity_type': 'EMAIL_ADDRESS',
            'start': 500,
            'end': 523,
            'score': 0.99
        },
        {
            'text': '0412 345 678',
            'entity_type': 'PHONE_NUMBER',
            'start': 600,
            'end': 612,
            'score': 0.90
        }
    ]


@pytest.fixture
def mock_presidio_analyzer():
    """Mock Presidio analyzer for testing."""
    from unittest.mock import Mock
    
    analyzer = Mock()
    analyzer.analyze.return_value = []
    return analyzer


@pytest.fixture
def mock_presidio_anonymizer():
    """Mock Presidio anonymizer for testing."""
    from unittest.mock import Mock
    
    anonymizer = Mock()
    anonymizer.anonymize.return_value = Mock(text="[ANONYMIZED TEXT]")
    return anonymizer


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing dataset processing."""
    return pd.DataFrame({
        'call_id': ['call_001', 'call_002'],
        'call_transcript': [
            'Hello, my name is John Smith and my email is john@example.com',
            'Hi, I am Jane Doe, you can reach me at jane.doe@test.com'
        ],
        'member_first_name': ['John', 'Jane'],
        'member_full_name': ['John Smith', 'Jane Doe'],
        'member_email': ['john@example.com', 'jane.doe@test.com'],
        'consultant_first_name': ['Sarah', 'Mike']
    })


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file for testing using a secure temporary file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        sample_csv_data.to_csv(temp_file.name, index=False)
        temp_path = temp_file.name
    
    yield temp_path
    
    # Clean up the temporary file after the test
    try:
        os.unlink(temp_path)
    except (OSError, PermissionError):
        pass  # Ignore cleanup errors


# Test markers setup
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "baseline: mark test for baseline implementations"
    )
    config.addinivalue_line(
        "markers", "agentic: mark test for agentic implementations"
    )
    config.addinivalue_line(
        "markers", "evaluation: mark test for evaluation logic"
    ) 