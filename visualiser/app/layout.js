import "./globals.css";
import Nav from "./components/Nav";

export const metadata = {
  title: "Spectral Clustering Benchmark",
  description:
    "COMP702 dissertation — benchmarking spectral clustering algorithms across datasets",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-ink text-[#e6e8ee] font-sans antialiased">
        <Nav />
        <main className="mx-auto max-w-6xl px-5 py-8">{children}</main>
        {/* <footer className="border-t border-edge text-muted text-[13px] text-center py-6">
          COMP702 Dissertation
        </footer> */}
      </body>
    </html>
  );
}
