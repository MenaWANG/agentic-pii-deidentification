# Plan B Implementation: Separate TextNormaliser + PurePresidioFramework

 

## Overview

Keep TextNormaliser and PurePresidioFramework as separate components. Add normalization as a preprocessing step and enhance visualization to show the three-stage workflow: Raw → Normalize → Clean.

 

## Implementation Steps

 

### Phase 1: Core Architecture (No Breaking Changes)

- [x] **Verify existing components work independently**
- [x] Test `TextNormaliser.normalize_text()` works as expected
- [x] Test `PurePresidioFramework.process_transcript()` works as expected
- [x] Confirm no changes needed to existing classes 

### Phase 2: Data Structure & Helper Functions

- [ ] **Create helper/wrapper functions**
- [ ] Create `process_transcript_with_normalization()` function in utils or new module
- [ ] Function should take `raw_text` and return dict with `raw_text`, `normalized_text`, `anonymized_text`, `detected_pii`
- [ ] Consider what's the best naming that will works best with other modules, and across different situations
        - Do we always do normalization? (Maybe not, what if it is a online chat that we want to dedact)
        - If so, maybe keep "original transcrip" naming in metrics
        - Let process_transcript() & html_table functions to handle the naming differences
- [ ] Add timing metrics for normalization and presidio steps separately
- [ ] Test the wrapper function with sample data
 

- [ ] **Update data structures**
- [ ] Ensure result dictionaries include both `raw_text` and `normalized_text`
- [ ] Maintain backward compatibility with existing result structure
- [ ] Document the new data structure format
 

### Phase 3: Visualization Enhancement

- [ ] **Create `create_three_stage_html_table()` function**
- [ ] Add function to `diagnostics.py`
- [ ] When `show_raw_transcript=False`: calls `create_diagnostic_html_table_configurable()` with `normalized_text` as "Transcript"
- [ ] When `show_raw_transcript=True`: adds Raw Transcript column to the left
- [ ] Test both modes work correctly
- [ ] Ensure all existing highlighting logic works (TP, FP, missed PII)
 

- [ ] **Test the HTML table function**
- [ ] Create test data with `raw_text`, `normalized_text`, `anonymized_text`
- [ ] Test `show_raw_transcript=False` mode (should look like existing table)
- [ ] Test `show_raw_transcript=True` mode (should show 4 columns)
- [ ] Verify styling and layout work correctly

 

### Phase 4: Demo Integration

- [ ] **Update demo notebooks**
- [ ] Create example showing Raw → Normalize → Clean workflow
- [ ] Use `TextNormaliser` + `PurePresidioFramework` separately in demos
- [ ] Show `create_three_stage_html_table()` with both modes
- [ ] Add examples demonstrating normalization benefits
 

- [ ] **Create convenience demo functions**
- [ ] `show_normalization_examples()`: Show before/after normalization
- [ ] `run_three_stage_demo()`: Complete workflow demonstration
- [ ] Functions should use existing test data from test suite

 

### Phase 5: Documentation & Testing

- [ ] **Update documentation**
- [ ] Update README with three-stage workflow examples
- [ ] Document the new helper functions
- [ ] Add examples of using components separately vs together

 

- [ ] **Add tests**

- [ ] Test helper/wrapper functions
- [ ] Test `create_three_stage_html_table()` function
- [ ] Test integration between TextNormaliser and PurePresidioFramework
- [ ] Ensure all existing tests still pass (no breaking changes)

 

### Phase 6: Validation & Cleanup

- [ ] **End-to-end testing**

- [ ] Run complete workflow on test data
- [ ] Verify performance metrics are accurate
- [ ] Test with both business and research matching modes
- [ ] Validate HTML output in notebooks

 

- [ ] **Code cleanup**

- [ ] Remove any unused imports or functions
- [ ] Add proper type hints to new functions
- [ ] Ensure consistent naming conventions
- [ ] Add docstrings to all new functions

 

## Key Design Principles

- ✅ **No breaking changes**: All existing code continues to work
- ✅ **Separation of concerns**: Each class has single responsibility
- ✅ **Backward compatibility**: Existing APIs unchanged
- ✅ **Flexible usage**: Can use normalization independently or skip it
- ✅ **Clear pipeline**: Raw → Normalize → Clean is explicit

 

## Success Criteria

- [ ] All existing tests pass without modification
- [ ] Can demonstrate three-stage workflow in demo notebooks
- [ ] HTML table shows raw transcript when requested
- [ ] Performance evaluation works on normalized text (as before)
- [ ] No import changes needed in existing files
- [ ] Zero breaking changes to existing functionality

 

## Files to Modify (Minimal Changes)

- [ ] `src/evaluation/diagnostics.py` - Add `create_three_stage_html_table()`
- [ ] `src/utils/` or new module - Add helper functions
- [ ] `demo/` notebooks - Update to show three-stage workflow
- [ ] `tests/` - Add tests for new functions
- [ ] `README.md` - Update with new examples

 

## Files NOT to Modify (Keep Unchanged)

- ✅ `src/baseline/presidio_framework.py` - Keep PurePresidioFramework as-is
- ✅ `src/utils/text_normaliser.py` - Keep TextNormaliser as-is
- ✅ `src/evaluation/metrics.py` - Keep evaluation logic unchanged
- ✅ All existing test files - Should pass without changes