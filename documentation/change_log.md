# Changelog

## [Unreleased]
### Added
- **Improved Logging**:
  - Enhanced logging across multiple modules to include debug-level details for key processing steps, data validation, and model training.
  - Debug-level logs provide additional context on data shapes, structures, and processing steps.
  - Updated in:
    - `train()` method in `models.ml_model.py`
    - `split_data()` function in `data.data_splitter.py`
    - Main script in `main.py`

- **Input Validation**:
  - Added robust input validation for `X_train`, `y_train`, `X_val`, and `y_val` in the `train()` method to ensure they conform to expected data types and shapes.
  - Validation checks include:
    - Ensuring `X_train` and `X_val` are 3D NumPy arrays.
    - Ensuring `y_train` and `y_val` are 2D arrays.
  - Updated in:
    - `train()` method in `models.ml_model.py`

### Changed
- **Reshaped `train()` Logic**:
  - Removed unnecessary reliance on Pandas-specific methods like `.columns` and `.head()` in favor of a streamlined approach that handles `numpy.ndarray` directly.
  - Simplified input data reshaping logic by assuming data from `split_data()` is already in the correct 3D format.
  - Updated in:
    - `train()` method in `models.ml_model.py`

- **Error Messages**:
  - Updated error handling in `train()` to provide more informative messages, making debugging easier.
  - Example: Error message now specifies when input dimensions are incorrect (e.g., "`X_train must be 3D but got shape {X_train.shape}`").
  - Updated in:
    - `train()` method in `models.ml_model.py`
    - Main script in `main.py`

- **Callbacks Handling**:
  - Adjusted `train()` to exclude the `ModelCheckpoint` callback when validation data is not provided.
  - Updated in:
    - `train()` method in `models.ml_model.py`

### Fixed
- **Training Errors**:
  - Resolved the `'numpy.ndarray' object has no attribute 'columns'` issue by ensuring the code properly processes NumPy arrays.
  - Ensured reshaped data maintains compatibility with the model input.
  - Fixed in:
    - `train()` method in `models.ml_model.py`

- **Data Validation in `split_data`**:
  - Added a check for missing keys (`'X'` or `'y'`) in the input data for each ticker.
  - Improved logging of data shapes and structure during the data splitting process.
  - Fixed in:
    - `split_data()` function in `data.data_splitter.py`

### Documentation
- Added detailed docstrings to clarify functionality and expected input formats.
  - Updated in:
    - `train()` method in `models.ml_model.py`
    - `split_data()` function in `data.data_splitter.py`

### Files Updated
1. **`main.py`**:
   - Enhanced error handling and logging.
   - Improved debugging for the processing of tickers and validation of data shapes.
2. **`models.ml_model.py`**:
   - Updated the `train()` method with improved validation, reshaping logic, and error messages.
   - Enhanced logging for debugging and error tracking.
3. **`data.data_splitter.py`**:
   - Updated `split_data()` function to include checks for missing data and improved logging.

