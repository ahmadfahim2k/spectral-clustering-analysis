import Link from "next/link";

const ALGOS = [
  ["K-Means", "baseline; Lloyd (1957)"],
  ["Unnormalised SC", "L = D − W (von Luxburg, 2007)"],
  ["Shi-Malik SC", "random-walk normalised cut (2000)"],
  ["Ng-Jordan-Weiss SC", "symmetric normalised (2001)"],
  ["Self-Tuning SC", "local scaling (Zelnik-Manor & Perona, 2004)"],
];

function Stat({ value, label }) {
  return (
    <div className="rounded-xl border border-edge bg-panel px-5 py-4">
      <div className="text-3xl font-bold">{value}</div>
      <div className="text-muted text-sm mt-1">{label}</div>
    </div>
  );
}

export default function OverviewPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold">Benchmarking Spectral Clustering Algorithms</h1>
      <p className="text-muted max-w-3xl mt-2">
        Visualizer for inspecting both the aggregate metric scores and the individual cluster assignments.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <Stat value="5" label="clustering algorithms" />
        <Stat value="3" label="synthetic shapes (rings, moons, blobs)" />
        <Stat value="3" label="real-world networks" />
        <Stat value="3" label="evaluation embeddings" />
      </div>

      <h2 className="text-lg font-semibold mt-9 mb-3">Algorithms</h2>
      <div className="grid md:grid-cols-2 gap-3">
        {ALGOS.map(([name, desc]) => (
          <div key={name} className="rounded-xl border border-edge bg-panel px-4 py-3">
            <span className="font-semibold">{name}</span>
            <span className="text-muted text-sm"> — {desc}</span>
          </div>
        ))}
      </div>

      <h2 className="text-lg font-semibold mt-9 mb-3">Explore</h2>
      <div className="grid md:grid-cols-2 gap-4">
        <Link
          href="/dashboard"
          className="rounded-xl border border-edge bg-panel px-5 py-5 hover:border-accent transition-colors"
        >
          <div className="font-semibold">Results Dashboard →</div>
          <p className="text-muted text-sm mt-1.5">
            A colour-coded heatmap of every algorithm × dataset score, with a metric selector
            (ARI/NMI for synthetic; Silhouette, Davies-Bouldin, Modularity for the networks).
            Degenerate results are flagged with a warning marker.
          </p>
        </Link>
        <Link
          href="/explorer"
          className="rounded-xl border border-edge bg-panel px-5 py-5 hover:border-accent transition-colors"
        >
          <div className="font-semibold">Dataset Explorer →</div>
          <p className="text-muted text-sm mt-1.5">
            Pick a dataset and algorithm to see the resulting clusters — a D3 scatter plot for
            synthetic data, a force-directed graph for the networks (nodes coloured by cluster,
            edges = routes / interactions / friendships) — with a metrics panel.
          </p>
        </Link>
      </div>

      <h2 className="text-lg font-semibold mt-9 mb-3">Methodology</h2>
      <p className="text-muted max-w-3xl">
        Similarity graphs use a Gaussian RBF kernel with the median-heuristic bandwidth for
        synthetic data, and observed edges for the networks. The number of clusters <em>k</em> is
        chosen by a two-step method (eigengap shortlist + downstream quality). Networks have no
        ground truth, so cluster quality is judged by internal indices under three embeddings
        (unnormalised / symmetric-normalised Laplacian, and shortest-path MDS) plus modularity.
        Every run uses a fixed random seed for reproducibility.
      </p>
    </div>
  );
}
