import { GITHUB_REPO_URL } from "@/lib/data";

export function SiteFooter() {
  return (
    <footer className="border-t border-border/60 px-6 py-10">
      <div className="mx-auto flex w-full max-w-[1080px] flex-col items-start gap-3 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <p>MIT-licensed. Dragos Calin, 2026.</p>
        <nav className="flex flex-wrap gap-4">
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="transition-colors hover:text-foreground"
          >
            GitHub repository
          </a>
          <a href="#methodology" className="transition-colors hover:text-foreground">
            Methodology
          </a>
          <a
            href="https://www.linkedin.com/in/dragoscalin/"
            target="_blank"
            rel="noreferrer noopener"
            className="transition-colors hover:text-foreground"
          >
            LinkedIn
          </a>
        </nav>
      </div>
    </footer>
  );
}
