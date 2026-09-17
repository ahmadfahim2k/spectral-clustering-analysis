"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { PALETTE } from "../lib/data";

// Canvas force-directed view using PRECOMPUTED positions (from the export step),
// so there is no live simulation — fast even at 4,039 nodes / thousands of edges.
// Nodes coloured by cluster; edges thin and translucent.
export default function ForceGraph({ positions, edges, labels, size = 460 }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!positions || !positions.length) return;
    const canvas = ref.current;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const pad = 16;
    const xs = positions.map((p) => p[0]);
    const ys = positions.map((p) => p[1]);
    const x = d3.scaleLinear().domain(d3.extent(xs)).range([pad, size - pad]);
    const y = d3.scaleLinear().domain(d3.extent(ys)).range([size - pad, pad]);

    // background
    ctx.fillStyle = "#10131b";
    ctx.strokeStyle = "#2a2f3d";
    ctx.lineWidth = 1;
    roundRect(ctx, 0.5, 0.5, size - 1, size - 1, 10);
    ctx.fill();
    ctx.stroke();

    // edges
    ctx.strokeStyle = "rgba(154,162,180,0.10)";
    ctx.lineWidth = 0.4;
    ctx.beginPath();
    for (const [i, j] of edges || []) {
      ctx.moveTo(x(positions[i][0]), y(positions[i][1]));
      ctx.lineTo(x(positions[j][0]), y(positions[j][1]));
    }
    ctx.stroke();

    // nodes
    const r = positions.length > 2000 ? 1.8 : positions.length > 800 ? 2.4 : 3.2;
    for (let i = 0; i < positions.length; i++) {
      ctx.beginPath();
      ctx.fillStyle = PALETTE[(labels ? labels[i] : 0) % PALETTE.length];
      ctx.arc(x(positions[i][0]), y(positions[i][1]), r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [positions, edges, labels, size]);

  return <canvas ref={ref} style={{ width: "100%", height: "auto" }} />;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
