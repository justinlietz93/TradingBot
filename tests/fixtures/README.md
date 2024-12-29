# Test Fixtures

This directory contains test fixtures used in the trading bot test suite.

## Purpose

Test fixtures provide:
- Consistent test environments
- Reusable test data
- Mock objects and configurations
- Standard test scenarios

## Components

1. Data Fixtures
   - Market data samples
   - Feature sets
   - Model inputs/outputs

2. Model Fixtures
   - Trained model states
   - Model configurations
   - Prediction samples

3. Strategy Fixtures
   - Trading signals
   - Position states
   - Portfolio snapshots

4. Configuration Fixtures
   - Test configurations
   - Environment settings
   - Parameter sets

## Usage

```python
import pytest
from fixtures.market_data import sample_data
from fixtures.model_states import trained_model

def test_strategy(sample_data, trained_model):
    # Test implementation
    pass
```

## Guidelines

- Keep fixtures focused and minimal
- Document fixture dependencies
- Maintain fixture isolation
- Update fixtures with code changes
