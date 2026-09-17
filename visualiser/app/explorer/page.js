"use client";

import { useEffect, useMemo, useState } from "react";
import ScatterPlot from "../components/ScatterPlot";
import ForceGraph from "../components/ForceGraph";
import GeoMap from "../components/GeoMap";
import MetricsPanel from "../components/MetricsPanel";
import { loadDashboard, loadExplorer } from "../lib/data";

export default function ExplorerPage() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState(null);
  const [geo, setGeo] = useState(null);
  const [algo, setAlgo] = useState(null);
  const [showTruth, setShowTruth] = useState(false);
  const [airportView, setAirportView] = useState("geo"); // 'geo' | 'network'
  const [error, setError] = useState(null);

  // dataset list from the dashboard file
  useEffect(() => {
    loadDashboard()
      .then((d) => {
        setDatasets(d.datasets);
        setDatasetId(d.datasets[0]?.id ?? null);
      })
      .catch((e) => setError(e.message));
  }, []);

  // geometry for the selected dataset
  useEffect(() => {
    if (!datasetId) return;
    setGeo(null);
    loadExplorer(datasetId)
      .then((g) => {
        setGeo(g);
        setAlgo(Object.keys(g.algorithms)[0] ?? null);
        setShowTruth(false);
        setAirportView("geo"); // default to the geographic view for airports
      })
      .catch((e) => setError(e.message));
  }, [datasetId]);

  const meta = useMemo(() => datasets.find((d) => d.id === datasetId), [datasets, datasetId]);

  if (error) return <div className="text-muted py-10 text-center">Could not load: {error}</div>;
  if (!datasets.length) return <div className="text-muted py-10 text-center">Loading…</div>;

  const algoData = geo && algo ? geo.algorithms[algo] : null;
  const labels = geo
    ? showTruth && geo.groundTruth
      ? geo.groundTruth
      : algoData
      ? algoData.labels
      : null
    : null;
  const entry = algoData
    ? { algorithm: algo, k: algoData.k, flag: algoData.flag, clusterSizes: algoData.clusterSizes, metrics: algoData.metrics }
    : null;

  return (
    <div>
      <h1 className="text-2xl font-bold">Dataset Explorer</h1>
      <p className="text-muted max-w-3xl mt-2">
        Pick a dataset and algorithm to see the resulting clusters and metrics. Colours denote
        cluster membership.
      </p>

      <div className="flex flex-wrap gap-2 mt-5">
        {datasets.map((d) => (
          <button
            key={d.id}
            onClick={() => setDatasetId(d.id)}
            className={
              "rounded-full px-3.5 py-1.5 text-[13px] border transition-colors " +
              (d.id === datasetId
                ? "text-white border-accent bg-gradient-to-b from-accent to-accent2"
                : "text-muted border-edge bg-panel2 hover:text-white")
            }
          >
            {d.label}
          </button>
        ))}
      </div>

      {!geo ? (
        <div className="text-muted py-10 text-center">Loading dataset…</div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4 mt-4">
          <div className="rounded-xl border border-edge bg-panel p-4">
            <div className="flex flex-wrap items-end gap-4 mb-3">
              <label className="flex flex-col gap-1.5 text-[13px] text-muted">
                Algorithm
                <select
                  value={algo || ""}
                  onChange={(e) => setAlgo(e.target.value)}
                  className="bg-panel2 text-[#e6e8ee] border border-edge rounded-lg px-2.5 py-2 text-sm"
                >
                  {Object.keys(geo.algorithms).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
              {geo.hasGroundTruth && (
                <label className="flex items-center gap-2 text-[13px] text-muted">
                  <input
                    type="checkbox"
                    checked={showTruth}
                    onChange={(e) => setShowTruth(e.target.checked)}
                  />
                  Show ground truth
                </label>
              )}
              {datasetId === "airports" && (
                <div className="ml-auto inline-flex rounded-lg border border-edge overflow-hidden">
                  {[
                    ["geo", "Geographic"],
                    ["network", "Network"],
                  ].map(([v, lbl]) => (
                    <button
                      key={v}
                      onClick={() => setAirportView(v)}
                      className={
                        "px-3 py-1.5 text-[13px] transition-colors " +
                        (airportView === v
                          ? "bg-accent text-white"
                          : "bg-panel2 text-muted hover:text-white")
                      }
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {datasetId === "airports" && airportView === "geo" ? (
              <GeoMap latlon={geo.latlon} labels={labels} />
            ) : geo.type === "scatter" ? (
              <ScatterPlot points={geo.points} labels={labels} />
            ) : (
              <ForceGraph positions={geo.positions} edges={geo.edges} labels={labels} />
            )}

            <p className="text-muted text-[13px] mt-2">
              {datasetId === "airports" && airportView === "geo"
                ? `${geo.mappedCount ?? ""} of ${meta?.n} airports shown at real coordinates (unmatched omitted) · colour = cluster`
                : geo.type === "network"
                ? `${meta?.n} nodes · edges = ${
                    datasetId === "airports"
                      ? "flight routes"
                      : datasetId === "yeast"
                      ? "interactions"
                      : "friendships"
                  } (sample drawn)`
                : showTruth
                ? "Colours show the ground-truth clusters."
                : "Colours show the selected algorithm's predicted clusters."}
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <MetricsPanel entry={entry} hasGroundTruth={geo.hasGroundTruth} />
            {meta?.note && (
              <div className="rounded-xl border border-edge bg-panel p-4 text-sm text-muted">
                <span className="text-white font-medium">About this dataset. </span>
                {meta.note}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
