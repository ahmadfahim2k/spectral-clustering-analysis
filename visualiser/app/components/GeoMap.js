"use client";

import { useMemo } from "react";
import { PALETTE } from "../lib/data";

// Simplified continental-US outline ([lon, lat]); coarse but recognisable, used
// only as a faint basemap for orientation (no mapping library).
const US_OUTLINE = [
  [-124.7, 48.4], [-124.1, 46.3], [-124.0, 43.0], [-123.8, 40.0], [-122.4, 37.8],
  [-120.6, 34.6], [-117.3, 32.5], [-114.7, 32.7], [-111.1, 31.3], [-108.2, 31.8],
  [-106.5, 31.8], [-103.0, 29.0], [-101.4, 29.8], [-99.5, 27.5], [-97.4, 25.9],
  [-95.3, 29.0], [-93.8, 29.7], [-91.0, 29.2], [-89.2, 29.3], [-88.0, 30.4],
  [-85.6, 30.0], [-84.0, 30.1], [-82.9, 27.8], [-81.8, 25.9], [-80.9, 25.2],
  [-80.5, 27.5], [-81.4, 30.7], [-79.2, 33.8], [-76.3, 35.0], [-75.5, 37.0],
  [-74.0, 40.5], [-71.9, 41.3], [-70.7, 41.6], [-70.2, 43.6], [-67.0, 44.9],
  [-69.2, 47.4], [-71.5, 45.0], [-76.0, 43.6], [-79.0, 43.3], [-82.5, 41.7],
  [-83.1, 42.0], [-82.5, 45.0], [-84.8, 45.9], [-87.6, 45.1], [-88.0, 47.0],
  [-90.0, 46.7], [-95.0, 49.0], [-104.0, 49.0], [-116.0, 49.0], [-122.8, 49.0],
];

const LON = [-125, -66];
const LAT = [24, 50];
const W = 720;
const H = 397;
const PAD = 8;

const sx = (lon) => PAD + ((lon - LON[0]) / (LON[1] - LON[0])) * (W - 2 * PAD);
const sy = (lat) => (H - PAD) - ((lat - LAT[0]) / (LAT[1] - LAT[0])) * (H - 2 * PAD);

// Airports plotted at real lat/lon, coloured by cluster (mirrors 02_airports Section 9).
export default function GeoMap({ latlon, labels }) {
  const outlinePath = useMemo(
    () => US_OUTLINE.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ") + " Z",
    []
  );
  const pts = useMemo(() => {
    if (!latlon) return [];
    const out = [];
    for (let i = 0; i < latlon.length; i++) {
      const c = latlon[i];
      if (!c) continue;
      const [lat, lon] = c;
      if (lon < LON[0] || lon > LON[1] || lat < LAT[0] || lat > LAT[1]) continue;
      out.push({ x: sx(lon), y: sy(lat), c: PALETTE[(labels ? labels[i] : 0) % PALETTE.length] });
    }
    return out;
  }, [latlon, labels]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="10" fill="#10131b" stroke="#2a2f3d" />
      <path d={outlinePath} fill="#171a23" stroke="#39415a" strokeWidth="1" />
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3.4" fill={p.c} fillOpacity="0.9" stroke="#0c0f16" strokeWidth="0.4" />
      ))}
    </svg>
  );
}
