"use client";

import { metricColor, fmt } from "../lib/data";

// props: datasets [{id,label}], algorithms [name], resultIndex {`${dataset}|${algo}`: result}, metric
export default function Heatmap({ datasets, algorithms, resultIndex, metric }) {
  return (
    <div className="overflow-x-auto thin-scroll">
      <table className="border-collapse w-full text-sm">
        <thead>
          <tr>
            <th className="text-left text-muted font-semibold p-2.5 sticky left-0 bg-ink">
              Algorithm
            </th>
            {datasets.map((d) => (
              <th key={d.id} className="text-muted font-semibold p-2.5 min-w-[110px]">
                {d.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {algorithms.map((algo) => (
            <tr key={algo}>
              <td className="text-left whitespace-nowrap p-2.5 sticky left-0 bg-ink border-t border-edge">
                {algo}
              </td>
              {datasets.map((d) => {
                const r = resultIndex[`${d.id}|${algo}`];
                const v = r ? r.metrics[metric.field] : null;
                const color = metricColor(v, metric);
                const isNA = v === null || v === undefined || Number.isNaN(v);
                const flagged = r && r.flag;
                return (
                  <td
                    key={d.id}
                    className="p-1 border-t border-edge text-center"
                    title={
                      flagged
                        ? `${algo} on ${d.label}: ${r.flag}`
                        : r
                        ? `${algo} on ${d.label}: sizes [${r.clusterSizes.join(", ")}]`
                        : ""
                    }
                  >
                    <div
                      className="relative rounded-md py-2 px-1 font-semibold tabular-nums"
                      style={{
                        background: isNA ? "#1e2230" : color,
                        color: isNA ? "#6b7280" : "#10141c",
                      }}
                    >
                      {isNA ? "N/A" : fmt(v)}
                      {flagged && (
                        <span
                          className="absolute -top-1.5 -right-1.5 text-[11px] leading-none rounded-full bg-amber-400 text-black w-4 h-4 flex items-center justify-center border border-black/40"
                          aria-label="degenerate result"
                        >
                          ⚠
                        </span>
                      )}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
