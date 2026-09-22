# Data Analysis using Spectral Clustering Algorithms — Source Code

This archive contains the source code used to produce the results and
analysis presented in the accompanying dissertation.

**Live demo:** the visualisation tool is deployed at
https://spectral-clustering-analysis.vercel.app/ — no setup required to explore
the results in a browser. To run it locally, see "Visualisation Tool" below.

## Project Structure

```
scripts/            Core Python package: algorithms, graph construction,
                    k-selection, evaluation metrics, dataset loaders, and the
                    data-export/validation helper scripts
notebooks/          Jupyter notebooks: one per dataset plus a metric-validation
                    study, containing the full analysis and diagnostics
data/               Processed/cached datasets (see "Datasets" below)
results/            JSON (and one .npz cache) output produced by the notebooks
visualiser/         Next.js browser-based visualisation tool
requirements.txt    Python dependencies
README.md           This file
```

## Requirements

- Python 3.10+
- Node.js 18+ (for the visualiser only)

## Python Setup

1. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run the notebooks:
   ```
   jupyter notebook
   ```
   Open and run, in order:
   - `notebooks/01_synthetic.ipynb`
   - `notebooks/02_airports.ipynb`
   - `notebooks/03_yeast.ipynb`
   - `notebooks/04_facebook.ipynb`

   Then, optionally (a standalone appendix study, no ordering dependency):
   - `notebooks/k_metric_validation.ipynb`

   Each dataset notebook writes its results to `results/*.json`.

## Datasets

- **Synthetic** — generated programmatically; no external file needed.
- **Yeast PPI** — `data/bio-yeast.mtx` (included; source: NetworkRepository).
- **Facebook Social Circles** — `data/facebook_combined.txt` (included;
  source: Stanford SNAP).
- **US Airports** — the raw Kaggle source file (`Airports2.csv`, ~500 MB) is
  **not included** in this archive due to its size. The aggregated graph used in
  the analysis is included as `data/airports_graph.csv`, together with
  `data/airports_coords.csv` (airport latitude/longitude, used by the
  visualiser's geographic map). To regenerate the aggregated files from scratch,
  download the raw dataset from
  https://www.kaggle.com/datasets/flashgordon/usa-airport-dataset,
  place `Airports2.csv` in `data/`, and run:
  ```
  python -c "from scripts.network_datasets import build_airport_graph, build_airport_coords; build_airport_graph(); build_airport_coords()"
  ```

## Visualisation Tool

The `visualiser/` directory contains a Next.js application (React + Tailwind CSS + D3.js) that reads pre-computed JSON and presents it interactively.

A hosted version is live at https://spectral-clustering-analysis.vercel.app/. To
run it locally instead:

1. Install dependencies:
   ```
   cd visualiser
   npm install
   ```
2. Run the development server:
   ```
   npm run dev
   ```
3. Open http://localhost:3000 in a browser.

The visualiser reads static data from `visualiser/public/data/` (already
generated and included in this archive), so it runs without any backend and does
not recompute anything live. To regenerate that data after re-running the
notebooks:
```
python -m scripts.export_visualiser_data
```

## Notes

- All experiments use a fixed random seed for reproducibility.
- `notebooks/04_facebook.ipynb` caches its five 4,039-node spectral
  decompositions to `results/facebook_cache.{json,npz}` (included) for fast
  re-execution; delete the cache to recompute from scratch.
- The five-algorithm k-voting study (dissertation appendix) can be run
  independently:
  ```
  python -m scripts.k_voting
  ```
