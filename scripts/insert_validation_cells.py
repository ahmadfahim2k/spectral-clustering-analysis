"""
One-shot helper: fold the Silhouette-vs-ARI validation into 01_synthetic.ipynb
as section 3b, without disturbing the existing (already-executed) cells.

Why this exists: the validation was first delivered as the standalone notebook
`notebooks/01b_silhouette_validation.ipynb`. This script inserts the same three
cells (markdown intro -> code -> interpretation) directly into
`notebooks/01_synthetic.ipynb`, just before "## 4. Run every algorithm", so the
main notebook is self-contained. Existing cells and their outputs are preserved.

Usage (from the repo root):

    python scripts/insert_validation_cells.py
    # then re-run the notebook to populate the new cell's output, e.g.:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_synthetic.ipynb

The script is idempotent — running it again detects the cells are already present
and does nothing.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "01_synthetic.ipynb"
MARKER = "silhouette_vs_ari"          # idempotency sentinel
ANCHOR_PREFIX = "## 4. Run every algorithm"

INTRO_MD = """## 3b. Validation — is Silhouette-based k-selection trustworthy?

On the real-world datasets (Airports, Yeast PPI, Facebook) there is **no ground
truth**, so the two-step selection must disambiguate candidate *k* with the
**Silhouette Score** rather than ARI. Before relying on it there, we validate it
here on the synthetic data, where the true *k* is known: we rerun the *same*
two-step selection but swap the criterion to Silhouette (`ground_truth=None`),
and compare the chosen *k* against both the true *k* and the ARI-based choice.

Silhouette is computed on the standardised feature matrix, exactly as
`select_spectral_k` does when no labels are supplied. This is a **diagnostic
only** — ARI remains the primary criterion everywhere else in this notebook and
in `scripts.run_synthetic`."""

CODE = """from scripts.validate_kselection import silhouette_vs_ari

# reruns the two-step selection with ARI and with Silhouette, prints the table
results = silhouette_vs_ari()"""

INTERP_MD = """### Interpretation

Read the `match?` column in the table above.

- **Blobs (convex, isotropic).** Silhouette measures within-cluster versus
  nearest-other-cluster distance in feature space, so it rewards compact,
  well-separated clusters. Blobs are exactly that, so Silhouette agrees with ARI
  and the true *k* = 3 here — Silhouette is reliable for this kind of structure.
- **Rings / moons (non-convex).** The *correct* partition groups points into
  interleaved, non-compact clusters (concentric rings; nested crescents). In
  Euclidean feature space those clusters are not compact, so Silhouette can score
  the true grouping *below* a more compact but wrong split — wherever the
  `match?` column shows a disagreement, this is the cause.

**Limitation to discuss in the dissertation.** Silhouette is a *biased* internal
criterion: it presupposes convex, compact clusters and therefore cannot be
trusted to select *k* for non-convex structure the way ARI can when labels exist.
Two consequences for the real-world datasets:

1. Where ground truth is unavailable, a Silhouette-selected *k* should be treated
   as a **guide, not ground truth** — cross-check it against the eigengap
   candidate shortlist and, where feasible, cluster stability or modularity.
2. **The space matters.** On the networks, Silhouette will be computed on the
   *spectral embedding* (there is no raw feature space), where well-separated
   communities are far more compact than concentric rings are in the plane. So
   Silhouette is likely *more* reliable there than this worst-case synthetic test
   suggests — but the convexity bias still applies, so the confidence ARI gives
   us here does not transfer automatically."""


def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)

    if any(MARKER in "".join(c.get("source", "")) for c in nb.cells):
        print("Validation cells already present — nothing to do.")
        return 0

    insert_at = next(
        (i for i, c in enumerate(nb.cells)
         if c.cell_type == "markdown" and "".join(c["source"]).lstrip().startswith(ANCHOR_PREFIX)),
        None,
    )
    if insert_at is None:
        print(f"Could not find the anchor cell ('{ANCHOR_PREFIX}'). No changes made.")
        return 1

    new_cells = [new_markdown_cell(INTRO_MD), new_code_cell(CODE), new_markdown_cell(INTERP_MD)]
    nb.cells[insert_at:insert_at] = new_cells
    nbformat.write(nb, NB_PATH)
    print(f"Inserted 3 validation cells at index {insert_at} "
          f"(before '{ANCHOR_PREFIX}...') in {NB_PATH.name}.")
    print("Now execute the notebook to populate the new output, e.g.:")
    print("  jupyter nbconvert --to notebook --execute --inplace notebooks/01_synthetic.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
