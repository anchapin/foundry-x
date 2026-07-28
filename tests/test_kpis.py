"""KPI tests — re-exported from observability/ modules (issue #1282).

This file is kept for backwards compatibility. All tests have been moved to:
- tests/observability/test_cycle_time_kpi.py
- tests/observability/test_regression_rate_kpi.py
- tests/observability/test_improvement_rate_kpi.py
- tests/observability/test_context_efficiency_kpi.py
- tests/observability/test_token_budget_kpi.py
- tests/observability/test_kpi_comparison.py
- tests/observability/test_kpi_cli.py
"""

# Re-export all test functions from the split modules for backwards compatibility
from tests.observability.test_context_efficiency_kpi import *
from tests.observability.test_cycle_time_kpi import *
from tests.observability.test_improvement_rate_kpi import *
from tests.observability.test_kpi_cli import *
from tests.observability.test_kpi_comparison import *
from tests.observability.test_regression_rate_kpi import *
from tests.observability.test_token_budget_kpi import *
