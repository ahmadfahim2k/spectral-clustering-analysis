/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export: the app is pure client-side and reads pre-generated JSON from
  // public/data/ (dashboard.json + explorer/*.json), so it can be hosted as plain
  // static files with no server (Vercel, Netlify, GitHub Pages, etc.).
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
