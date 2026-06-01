import Link from "next/link";

const NAV = [
  { href: "#findings", label: "Findings" },
  { href: "#curve", label: "Curve" },
  { href: "#scenarios", label: "Scenarios" },
  { href: "#sectors", label: "Sectors" },
  { href: "#validation", label: "Validation" },
  { href: "#methodology", label: "Methodology" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 max-w-[1080px] items-center justify-between px-6">
        <Link
          href="#top"
          className="text-sm font-medium tracking-tight text-foreground"
        >
          AI Labor Market Impact Observatory
        </Link>
        <nav className="hidden gap-6 text-xs text-muted-foreground md:flex">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="transition-colors hover:text-foreground"
            >
              {item.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
