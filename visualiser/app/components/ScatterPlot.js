"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { PALETTE } from "../lib/data";

// 2-D scatter of synthetic points, coloured by cluster label.
export default function ScatterPlot({ points, labels, size = 460 }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!points || !points.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const m = 14;
    const w = size - 2 * m;

    const x = d3.scaleLinear().domain(d3.extent(points, (p) => p[0])).nice().range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(points, (p) => p[1])).nice().range([w, 0]);

    const g = svg
      .attr("viewBox", `0 0 ${size} ${size}`)
      .append("g")
      .attr("transform", `translate(${m},${m})`);

    g.append("rect")
      .attr("width", w)
      .attr("height", w)
      .attr("rx", 10)
      .attr("fill", "#10131b")
      .attr("stroke", "#2a2f3d");

    g.selectAll("circle")
      .data(points)
      .join("circle")
      .attr("cx", (d) => x(d[0]))
      .attr("cy", (d) => y(d[1]))
      .attr("r", 3.6)
      .attr("fill", (_, i) => PALETTE[(labels ? labels[i] : 0) % PALETTE.length])
      .attr("fill-opacity", 0.9)
      .attr("stroke", "#0c0f16")
      .attr("stroke-width", 0.4);
  }, [points, labels, size]);

  return <svg ref={ref} className="w-full h-auto" />;
}
