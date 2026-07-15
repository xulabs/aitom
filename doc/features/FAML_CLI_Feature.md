## Feature: FAML Subtomogram Averaging CLI (`faml run`)

### 1. What Was Added

- **New CLI script:** `aitom/bin/faml.py`
  - Exposes a `faml` console command with a `run` subcommand.
- **Console entry in `setup.py`:**
  - `faml = aitom.bin.faml:main`
- **New test:** `tests/test_faml_cli.py`
  - Creates a tiny synthetic dataset and checks that `faml run` produces at least one checkpoint when the tomominer core is available (otherwise the test is skipped).

### 2. Why It Was Added

FAML is one of AITom’s key subtomogram averaging methods. The core EM implementation (`aitom/average/ml/faml/faml.py`) is already there and used in the tutorial `doc/tutorials/009_faml.py`, but:

- There was **no simple command-line entry** to run FAML on an existing dataset.
- Users had to write custom Python scripts to:
  - Load the LSM database.
  - Load the `dj.pickle` key file.
  - Call `faml.EM` with the right arguments.

This CLI turns the FAML EM pipeline into a single, reproducible command, making it consistent with other tools like `picking` and `disca`.

### 3. How It Works

**Command:**

```bash
faml run \
  --db data/aitom_demo_subtomograms.db \
  --dj data/dj.pickle \
  --K 2 \
  --iterations 20 \
  --snapshot-interval 5 \
  --out-dir faml_output
```

**Required inputs:**

- `--db`:
  - Path to the LSM database (`.db`) that stores:
    - Fourier-transformed subtomograms.
    - Corresponding masks (e.g., missing wedge masks).
  - This is the same format produced in `doc/tutorials/009_faml.py`.

- `--dj`:
  - Path to `dj.pickle` – a list of dictionaries describing which keys in the database belong to which subtomogram and mask, e.g.:
    ```python
    [
      {"v": "uuid_for_volume_fft", "m": "uuid_for_mask", "id": "5T2C"},
      {"v": "uuid_for_volume_fft", "m": "uuid_for_mask", "id": "1KP8"},
      ...
    ]
    ```

**Key parameters:**

- `--K` (required): number of classes/clusters to average (e.g., 2 for two structural classes).
- `--iterations` (default 20): number of EM iterations.
- `--snapshot-interval` (default 5): how often to save intermediate checkpoints and images.
- `--out-dir` (default `faml_output`): output directory for checkpoints and average volumes.
- `--reg`: enable regularization between class averages.
- `--no-voronoi`: disable Voronoi weighting of transformation configurations.

**What the CLI does internally:**

1. Loads the `dj.pickle` and LSM database path into a small `img_data` dict:
   ```python
   img_data = {"db_path": db_path, "dj": dj}
   ```
2. Calls the existing FAML EM implementation:
   ```python
   faml.EM(
       img_data=img_data,
       K=args.K,
       iteration=args.iterations,
       path=out_dir,
       snapshot_interval=args.snapshot_interval,
       reg=args.reg,
       use_voronoi=not args.no_voronoi,
   )
   ```
3. FAML:
   - Initializes class averages `A_k`.
   - Alternates between:
     - Estimating optimal transformations (alignment).
     - Updating class probabilities `alpha`, noise variance `sigma_sq`, translation variance `xi`.
     - Updating averages `A_k`.
   - Writes checkpoints and final results into `--out-dir`.

**Outputs:**

In `--out-dir`:

- `checkpoints/00000000.pickle`, `00000001.pickle`, …:
  - Contain `theta` dictionaries with the current state of the EM parameters (averages, variances, etc.).
- `kaverage.pickle`, `theta.pickle` (final outputs):
  - Per-class averaged volumes.
  - Final EM parameters.
- PNG images of slice grids (via `output_images`) for quick inspection.

### 4. How It Is Used in a Real AITom Workflow

Typical pipeline using FAML:

1. **Prepare subtomograms and masks**
   - Extract subtomograms (e.g., after particle picking).
   - Generate missing wedge masks (e.g., using `wedge_mask`).

2. **Build LSM database and dj.pickle** (as in `009_faml.py`):
   ```python
   from aitom.io.db.lsm_db import LSM
   from aitom.image.vol.wedge.util import wedge_mask
   import uuid, pickle

   image_db = LSM('data/aitom_demo_subtomograms.db')
   dj = []
   m = wedge_mask([32, 32, 32], ang1=30, sphere_mask=True, verbose=False)

   for s in subtomograms['5T2C_data']:
       v = fourier_transform(s)
       v_key = str(uuid.uuid4())
       m_key = str(uuid.uuid4())
       image_db[v_key] = v
       image_db[m_key] = m
       dj.append({'v': v_key, 'm': m_key, 'id': '5T2C'})

   with open('data/dj.pickle', 'wb') as f:
       pickle.dump(dj, f, protocol=-1)
   ```

3. **Run FAML averaging via CLI:**
   ```bash
   faml run \
     --db data/aitom_demo_subtomograms.db \
     --dj data/dj.pickle \
     --K 2 \
     --iterations 20 \
     --snapshot-interval 5 \
     --out-dir faml_output
   ```

4. **Inspect averages and use in downstream analysis:**
   - Look at PNG slice images in `faml_output` to see averaged structures.
   - Use final average volumes (from `theta.pickle` / `kaverage.pickle`) as:
     - Improved templates for template matching.
     - Inputs to further classification or reconstruction steps.

### 5. Testing

**File:** `tests/test_faml_cli.py`

The test:

1. Constructs a tiny synthetic dataset:
   - Creates a temporary LSM database with 4 random subtomograms of shape `(16, 16, 16)` and a single wedge mask.
   - Writes a matching `dj.pickle` with the expected `{'v', 'm', 'id'}` entries.
2. Tries to import `aitom.bin.faml.main`:
   - If the tomominer C++ core is not built and import fails, the test **skips** with a clear message.
3. If import succeeds, runs:
   ```python
   faml_main(
       [
           "run",
           "--db", str(db_path),
           "--dj", str(dj_path),
           "--K", "1",
           "--iterations", "1",
           "--snapshot-interval", "1",
           "--out-dir", str(out_dir),
       ]
   )
   ```
4. Asserts that at least one checkpoint pickle exists under `out_dir / "checkpoints"`.

This gives a small but realistic end-to-end check that:

- The CLI can wire parameters into `faml.EM`.
- FAML can run one EM iteration on a minimal dataset and produce output.
- The test does not break environments where the low-level core is not compiled – it simply skips in that case.


