/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export so the visualiser can be hosted as plain files alongside the
  // dissertation (e.g. GitHub Pages) with no server. Results are read from
  // /public/results/*.json, synced from the repo-root results/ folder.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
