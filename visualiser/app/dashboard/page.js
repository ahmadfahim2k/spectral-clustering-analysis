"use client";

import { useEffect, useMemo, useState } from "react";
import Heatmap from "../components/Heatmap";
import { loadDashboard, METRICS } from "../lib/data";

export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [metricKey, setMetricKey] = useState("silhouette");

  useEffect(() => {
    loadDashboard().then(setData).catch((e) => setError(e.message));
  }, []);

  const resultIndex = useMemo(() => {
    const idx = {};
    if (data) for (const r of data.results) idx[`${r.dataset}|${r.algorithm}`] = r;
    return idx;
  }, [data]);

  if (error) return <div className="text-muted py-10 text-center">Could not load results: {error}</div>;
  if (!data) return <div className="text-muted py-10 text-center">Loading results…</div>;

  const metric = METRICS.find((m) => m.key === metricKey);

  // count applicable cells so we can hint which datasets a metric covers
  const applies = (m) =>
    data.results.some((r) => {
      const v = r.metrics[m.field];
      return v !== null && v !== undefined && !Number.isNaN(v);
    });

  return (
    <div>
      <h1 className="text-2xl font-bold">Results Dashboard</h1>
      <p className="text-muted max-w-3xl mt-2">
        Every algorithm scored on every dataset. Green is better; the scale flips automatically
        for lower-is-better metrics. Cells that don&apos;t apply show <em>N/A</em>. A ⚠ marks a
        result the diagnostics found <span className="text-amber-300">degenerate</span> — a
        near-trivial split — so a deceptively high score isn&apos;t mistaken for a good clustering.
      </p>

      <div className="flex flex-wrap gap-2 mt-5">
        {METRICS.map((m) => (
          <button
            key={m.key}
            onClick={() => setMetricKey(m.key)}
            className={
              "rounded-full px-3.5 py-1.5 text-[13px] border transition-colors " +
              (m.key === metricKey
                ? "text-white border-accent bg-gradient-to-b from-accent to-accent2"
                : "text-muted border-edge bg-panel2 hover:text-white") +
              (applies(m) ? "" : " opacity-60")
            }
          >
            {m.label}
          </button>
        ))}
      </div>

      <p className="text-muted text-[13px] mt-3">
        Showing <span className="text-white font-medium">{metric.label}</span> ·{" "}
        {metric.higherBetter ? "higher is better" : "lower is better"}
        {(metric.key === "silhouette" || metric.key === "davies_bouldin") &&
          " · unnormalised-Laplacian embedding"}
        . ARI/NMI apply to synthetic datasets; Silhouette, Davies-Bouldin and Modularity to all.
      </p>

      <div className="rounded-xl border border-edge bg-panel p-4 mt-3">
        <Heatmap
          datasets={data.datasets}
          algorithms={data.algorithms}
          resultIndex={resultIndex}
          metric={metric}
        />
        <div className="flex items-center gap-2.5 text-muted text-xs mt-4">
          <span className="inline-block w-3.5 h-3.5 rounded" style={{ background: "#a50026" }} />
          worse
          <span className="inline-block w-3.5 h-3.5 rounded" style={{ background: "#ffffbf" }} />
          mid
          <span className="inline-block w-3.5 h-3.5 rounded" style={{ background: "#006837" }} />
          better
          <span className="ml-3 inline-flex items-center gap-1">
            <span className="text-[11px] rounded-full bg-amber-400 text-black w-4 h-4 flex items-center justify-center">
              ⚠
            </span>
            degenerate split (see Explorer for details)
          </span>
        </div>
      </div>
    </div>
  );
}
