## Feature: DISCA Clustering CLI (`disca cluster`)

### 1. What Was Added

- **New script:** `aitom/bin/disca.py`
- **New console entry:** in `setup.py`
  - `disca = aitom.bin.disca:main`

The CLI adds a `disca` command with a `cluster` subcommand:

```bash
disca cluster \
  --features features.npy \
  --candidate-k 5,10,20 \
  --out-labels disca_labels.npy \
  --out-summary disca_summary.json
```

### 2. Why It Was Added

DISCA’s pipeline has two stages:

1. A deep network (YOPO) that extracts feature vectors from subtomograms.
2. A clustering stage (Gaussian Mixture Models, multiple Ks) that assigns each sample to a structural class.

Previously, the **clustering logic** lived only inside research scripts. There was no:

- Simple **one-line interface** to run clustering on precomputed features.
- Clean way to integrate clustering into other tools or pipelines.

The CLI makes the clustering stage:

- Reusable from the shell or any workflow manager.
- Easy to script and automate (e.g., in bash or Snakemake).
- Clear and discoverable via `disca --help`.

### 3. How It Works (Exactly)

**Input:**

- `features.npy`: NumPy array with shape `(N, D)` (N samples, D-dimensional features).
  - Typically these are embeddings produced by DISCA’s YOPO network.

**Process:**

1. Loads `features.npy` into memory.
2. For each `K` in `--candidate-k` (e.g., `5,10,20`):
   - Fits a `GaussianMixture` model from `sklearn.mixture`.
   - Computes the BIC score on the same feature set.
3. Selects the **best K** as the one with **lowest BIC**.
4. Runs `gmm.predict(features)` to obtain integer labels.

**Output:**

- `disca_labels.npy` (configurable via `--out-labels`):
  - Shape `(N,)`, `int64`.
  - Cluster ID for each row of `features.npy`.
- `disca_summary.json` (configurable via `--out-summary`):
  - `chosen_k`: the selected K.
  - `bic`: BIC score for the selected model.
  - `counts`: per-cluster sample counts.
  - `candidate_k`, `reg_covar`, `max_iter`, `seed`: parameters used.

### 4. How It Fits a Real AITom Application

Typical workflow:

1. **Prepare subtomograms** using existing tools (picking + extraction).
2. **Extract DISCA embeddings** (existing DISCA scripts):
   - Run YOPO network over all subtomograms.
   - Save embeddings as `features.npy` (shape `(N, D)`).
3. **Cluster with CLI:**

   ```bash
   disca cluster \
     --features features.npy \
     --candidate-k 5,10,20 \
     --out-labels disca_labels.npy \
     --out-summary disca_summary.json
   ```

4. **Use labels in downstream analysis:**
   - Group subtomograms by label.
   - Run `average/simple_iterative` or `average/ml/faml` per cluster to get class averages.
   - Visualize class averages; evaluate clustering quality (FSC, etc.).

This turns the **clustering stage** of DISCA into a single, documented step that can be reused in pipelines and integrated with other AITom modules.

### 5. Validation / Testing

To validate behavior, we tested the CLI on synthetic data:

1. Generated 3 clear Gaussian clusters in 2D (N = 60, K = 3).
2. Saved them as `tmp_disca_test/features.npy`.
3. Ran:

   ```bash
   python -m aitom.bin.disca cluster \
     --features tmp_disca_test/features.npy \
     --candidate-k 2,3,4 \
     --out-labels tmp_disca_test/labels.npy \
     --out-summary tmp_disca_test/summary.json
   ```

4. Observed:
   - CLI chose **K = 3** (expected).
   - Cluster counts were ~20 samples per cluster.
   - Labels and summary saved successfully.

This shows the feature works correctly on a realistic clustering task and is safe to propose in a PR.


