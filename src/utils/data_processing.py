# This file contain utility functions for data loading and preparation. 

import pandas as pd
from typing import Dict, List, Union

def create_dynamic_field_mapping_for_row(
    row_data: Union[pd.Series, dict], 
    field_configs: Dict[str, Union[str, List[str]]]
) -> Dict[str, List[str]]:
    """
    Create a dynamic field mapping for a single row by filtering out null/None values from numbered fields.
    
    Args:
        row_data: Single row of data (pandas Series or dict)
        field_configs: Dictionary mapping PII tags to field patterns
                      e.g., {'<FIRST_NAME>': 'FIRSTNAME_', '<LAST_NAME>': 'SURNAME_'}
    
    Returns:
        Dictionary with PII tags mapped to lists of valid (non-null) field names for this specific row.
    
    Example:
        field_configs = {
            '<FIRST_NAME>': 'FIRST_NAME_',
            '<LAST_NAME>': 'SURNAME_'
        }
        
        # Sample DataFrame
        data = pd.DataFrame({
            'FIRST_NAME_FATHER': ['John', 'Kate'],
            'FIRST_NAME_MOTHER': ['Jane', None],
            'SURNAME_1': [None, 'Smith'],
            'SURNAME_2': ['Doe', 'Smith']
        })
        
        # For first row (index 0):
        # {
        #     '<FIRST_NAME>': ['FIRST_NAME_FATHER', 'FIRST_NAME_MOTHER'],
        #     '<LAST_NAME>': ['SURNAME_2']
        # }
        
        # For second row (index 1):
        # {
        #     '<FIRST_NAME>': ['FIRST_NAME_FATHER'],
        #     '<LAST_NAME>': ['SURNAME_1', 'SURNAME_2']
        # }
    """
    field_mapping = {}
    
    # Convert to dict if it's a pandas Series
    if hasattr(row_data, 'to_dict'):
        row_dict = row_data.to_dict()
    else:
        row_dict = row_data
    
    for pii_tag, field_pattern in field_configs.items():
        valid_fields = []
        
        # Handle special cases where field_pattern is a list (like phone numbers)
        if isinstance(field_pattern, list):
            for field_name in field_pattern:
                if field_name in row_dict:
                    value = row_dict[field_name]
                    if pd.notna(value) and value != 'None' and value != '':
                        valid_fields.append(field_name)
        else:
            # Find all columns matching the pattern in this specific row
            matching_fields = [col for col in row_dict.keys() if col.startswith(field_pattern)]
            
            # Check each matching field for non-null values in this row
            for field_name in matching_fields:
                value = row_dict[field_name]
                # Check if this specific row has a valid value for this field
                if pd.notna(value) and value != 'None' and value != '':
                    valid_fields.append(field_name)
        
        if valid_fields:
            field_mapping[pii_tag] = sorted(valid_fields)  # Sort for consistency
    
    return field_mapping