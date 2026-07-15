## Feature: DISCA CLI Synthetic Smoke Test

### 1. What Was Added

- **New test file:** `tests/test_disca_cli.py`
- **New dev dependency:** `pytest` added to `requirements.txt` for running tests.

This test is a **small, synthetic smoke test** that verifies the `disca cluster` command behaves correctly on a simple clustering problem.

### 2. Why It Was Added

The DISCA CLI provides a convenient way to run the clustering stage (GMM + BIC) on precomputed features. To keep this behavior stable across:

- Future code refactors,
- Dependency bumps (e.g., scikit-learn),
- Environment changes,

we want an automated check that confirms:

1. The CLI runs without error.
2. It selects a sensible number of clusters.
3. The outputs have correct shapes and internal consistency.

The smoke test gives maintainers a quick signal that the **core semantics** of the CLI are still intact.

### 3. How the Test Works

**File:** `tests/test_disca_cli.py`

Key steps inside the test:

1. **Create synthetic data**
   - Builds 3 well-separated Gaussian clusters in 2D:
     - 20 points near `[0, 0]`,
     - 20 points near `[3, 0]`,
     - 20 points near `[0, 3]`.
   - Concatenated into a `(60, 2)` NumPy array and saved to `features.npy` in a temporary directory (`tmp_path`).

2. **Run the CLI programmatically**
   - Adjusts `sys.path` so the `aitom` package is importable from the repo root.
   - Imports `main` from `aitom.bin.disca`.
   - Calls:
     ```python
     disca_main(
         [
             "cluster",
             "--features", str(features_path),
             "--candidate-k", "2,3,4",
             "--out-labels", str(labels_path),
             "--out-summary", str(summary_path),
         ]
     )
     ```

3. **Validate outputs**
   - Asserts that both `labels.npy` and `summary.json` were created.
   - Loads labels and summary and checks:
     - `labels.shape == (60,)`.
     - `summary["chosen_k"] == 3` (correct number of clusters).
     - The `counts` in the summary sum to 60 and contain exactly 3 clusters.

If any of these conditions fail, the test fails, signaling a possible regression.

### 4. How to Run the Test

From the repository root:

```bash
pip install -r requirements.txt  # ensures pytest and sklearn are available

pytest tests/test_disca_cli.py -q
```

Expected output:

```text
.                                                                        [100%]
1 passed, ... warnings in X.XXs
```

### 5. How This Helps the Project

- Provides a **fast, deterministic check** of the DISCA CLI on a controlled dataset.
- Catches regressions in:
  - The argument parsing and CLI wiring,
  - The GMM + BIC clustering logic,
  - The format and consistency of outputs.
- Can be integrated into future CI to ensure that changes to DISCA or its dependencies do not silently break the clustering interface.


