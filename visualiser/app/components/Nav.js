"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/dashboard", label: "Results Dashboard" },
  { href: "/explorer", label: "Dataset Explorer" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="sticky top-0 z-10 border-b border-edge bg-ink/90 backdrop-blur">
      <div className="mx-auto max-w-6xl px-5 py-3.5 flex items-center gap-6">
        <span className="font-bold tracking-tight">◑ Spectral Clustering Benchmark</span>
        <div className="ml-auto flex gap-5 text-sm">
          {LINKS.map((l) => {
            const active = path === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  active ? "text-white font-medium" : "text-muted hover:text-white transition-colors"
                }
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
