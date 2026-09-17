"use client";

import { fmt } from "../lib/data";

function Row({ name, value, na }) {
  return (
    <div className="flex justify-between py-2 border-b border-edge last:border-0">
      <span className="text-muted">{name}</span>
      <span className="font-semibold tabular-nums">{na ? "N/A" : value}</span>
    </div>
  );
}

export default function MetricsPanel({ entry, hasGroundTruth }) {
  if (!entry) return null;
  const m = entry.metrics || {};
  const isSynthetic = hasGroundTruth;

  return (
    <div className="rounded-xl border border-edge bg-panel p-4">
      <div className="flex items-center justify-between">
        <strong>{entry.algorithm}</strong>
        <span className="text-muted text-xs border border-edge rounded px-2 py-0.5">
          k = {entry.k}
        </span>
      </div>

      {entry.flag && (
        <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-200 text-sm px-3 py-2 flex gap-2">
          <span aria-hidden>⚠</span>
          <span>{entry.flag}</span>
        </div>
      )}

      <div className="mt-3 text-sm">
        {isSynthetic && (
          <>
            <Row name="ARI" value={fmt(m.ARI)} na={m.ARI == null} />
            <Row name="NMI" value={fmt(m.NMI)} na={m.NMI == null} />
          </>
        )}
        <Row name="Modularity (Q)" value={fmt(m.modularity)} na={m.modularity == null} />
        <Row name="Silhouette (unnorm.)" value={fmt(m.silhouette_unnorm)} na={m.silhouette_unnorm == null} />
        <Row name="Silhouette (sym.)" value={fmt(m.silhouette_sym)} na={m.silhouette_sym == null} />
        <Row name="Silhouette (shortest-path)" value={fmt(m.silhouette_shortest_path)} na={m.silhouette_shortest_path == null} />
        <Row name="Davies-Bouldin (unnorm.)" value={fmt(m.davies_bouldin_unnorm)} na={m.davies_bouldin_unnorm == null} />
      </div>

      {m.modularityNote && (
        <div className="mt-2 text-[11px] text-muted italic">Q: {m.modularityNote}.</div>
      )}

      <div className="mt-3 text-xs text-muted">
        Cluster sizes: [{entry.clusterSizes.join(", ")}]
      </div>
    </div>
  );
}
