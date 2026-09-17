// Client-side loaders + metric metadata for the static JSON produced by
// scripts/export_visualiser_data.py (served from /public/data/).

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

// Cluster colour palette shared by the D3 views (high-contrast, matches notebooks)
export const PALETTE = [
  "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
  "#00ced1", "#f781bf", "#a65628", "#999999", "#66c2a5",
];

// Dashboard metric tabs. `field` indexes into a result's metrics object.
export const METRICS = [
  { key: "ARI", label: "ARI", field: "ARI", higherBetter: true, domain: [-0.1, 1] },
  { key: "NMI", label: "NMI", field: "NMI", higherBetter: true, domain: [0, 1] },
  { key: "silhouette", label: "Silhouette", field: "silhouette_unnorm", higherBetter: true, domain: [-1, 1] },
  { key: "davies_bouldin", label: "Davies-Bouldin", field: "davies_bouldin_unnorm", higherBetter: false, domain: [0, 4] },
  { key: "modularity", label: "Modularity", field: "modularity", higherBetter: true, domain: [0, 0.75] },
];

export async function loadDashboard() {
  const res = await fetch(`${BASE}/data/dashboard.json`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load dashboard.json");
  return res.json();
}

export async function loadExplorer(id) {
  const res = await fetch(`${BASE}/data/explorer/${id}.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load explorer/${id}.json`);
  return res.json();
}

// Green(good) -> yellow -> red(bad) for a metric value; honours higher/lower-better.
export function metricColor(value, metric) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const [lo, hi] = metric.domain;
  let t = (value - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  if (!metric.higherBetter) t = 1 - t; // low DB is good
  // t=0 red, 0.5 yellow, 1 green
  const stops = [
    [165, 0, 38],
    [215, 48, 39],
    [244, 109, 67],
    [253, 174, 97],
    [254, 224, 139],
    [217, 239, 139],
    [166, 217, 106],
    [102, 189, 99],
    [26, 152, 80],
    [0, 104, 55],
  ];
  const x = t * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(3);
}
