import pandas as pd
import pytest
from src.utils.data_processing import create_dynamic_field_mapping_for_row

def test_create_dynamic_field_mapping_family_example():
    """Test the exact example from the docstring with family member patterns."""
    # Setup test data exactly as in the docstring
    data = pd.DataFrame({
        'FIRST_NAME_FATHER': ['John', 'Kate'],
        'FIRST_NAME_MOTHER': ['Jane', None],
        'SURNAME_1': [None, 'Smith'],
        'SURNAME_2': ['Doe', 'Smith']
    })
    
    field_configs = {
        '<FIRST_NAME>': 'FIRST_NAME_',
        '<LAST_NAME>': 'SURNAME_'
    }
    
    # Test first row (index 0)
    row0_mapping = create_dynamic_field_mapping_for_row(data.iloc[0], field_configs)
    assert row0_mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_FATHER', 'FIRST_NAME_MOTHER'],
        '<LAST_NAME>': ['SURNAME_2']
    }
    
    # Test second row (index 1)
    row1_mapping = create_dynamic_field_mapping_for_row(data.iloc[1], field_configs)
    assert row1_mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_FATHER'],
        '<LAST_NAME>': ['SURNAME_1', 'SURNAME_2']
    }

def test_create_dynamic_field_mapping_mixed_patterns():
    """Test handling of mixed patterns - some with numbers, some with descriptive suffixes."""
    data = pd.DataFrame({
        'FIRST_NAME_FATHER': ['John', None],
        'FIRST_NAME_1': ['James', 'Kate'],
        'FIRST_NAME_MOTHER': [None, 'Jane'],
        'SURNAME_HOME': ['Smith', None],
        'SURNAME_WORK': ['Jones', 'Brown']
    })
    
    field_configs = {
        '<FIRST_NAME>': 'FIRST_NAME_',
        '<LAST_NAME>': 'SURNAME_'
    }
    
    # Test first row
    row0_mapping = create_dynamic_field_mapping_for_row(data.iloc[0], field_configs)
    assert row0_mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_1', 'FIRST_NAME_FATHER'],
        '<LAST_NAME>': ['SURNAME_HOME', 'SURNAME_WORK']
    }
    
    # Test second row
    row1_mapping = create_dynamic_field_mapping_for_row(data.iloc[1], field_configs)
    assert row1_mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_1', 'FIRST_NAME_MOTHER'],
        '<LAST_NAME>': ['SURNAME_WORK']
    }

def test_create_dynamic_field_mapping_empty_values():
    """Test handling of various empty/null values with descriptive patterns."""
    data = pd.DataFrame({
        'FIRST_NAME_FATHER': ['', 'John'],
        'FIRST_NAME_MOTHER': ['None', pd.NA],
        'SURNAME_HOME': [None, ''],
    })
    
    field_configs = {
        '<FIRST_NAME>': 'FIRST_NAME_',
        '<LAST_NAME>': 'SURNAME_'
    }
    
    # Test first row - should have no valid fields
    row0_mapping = create_dynamic_field_mapping_for_row(data.iloc[0], field_configs)
    assert row0_mapping == {}
    
    # Test second row - should only have FIRST_NAME_FATHER
    row1_mapping = create_dynamic_field_mapping_for_row(data.iloc[1], field_configs)
    assert row1_mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_FATHER']
    }

def test_create_dynamic_field_mapping_dict_input():
    """Test function works with dictionary input using descriptive patterns."""
    row_dict = {
        'FIRST_NAME_FATHER': 'John',
        'FIRST_NAME_MOTHER': 'Mary',
        'SURNAME_HOME': None,
        'OTHER_FIELD': 'irrelevant'
    }
    
    field_configs = {
        '<FIRST_NAME>': 'FIRST_NAME_',
        '<LAST_NAME>': 'SURNAME_'
    }
    
    mapping = create_dynamic_field_mapping_for_row(row_dict, field_configs)
    assert mapping == {
        '<FIRST_NAME>': ['FIRST_NAME_FATHER', 'FIRST_NAME_MOTHER']
    }

def test_create_dynamic_field_mapping_no_matches():
    """Test behavior when no fields match the patterns."""
    row_dict = {
        'FATHER_NAME': 'John',
        'MOTHER_NAME': 'Mary',
        'OTHER': 'data'
    }
    
    field_configs = {
        '<FIRST_NAME>': 'FIRST_NAME_',
        '<LAST_NAME>': 'SURNAME_'
    }
    
    mapping = create_dynamic_field_mapping_for_row(row_dict, field_configs)
    assert mapping == {} 