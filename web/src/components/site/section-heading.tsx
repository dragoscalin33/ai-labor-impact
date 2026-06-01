import { cn } from "@/lib/utils";

const ACCENTS = {
  default: { text: "text-muted-foreground", bar: "bg-foreground/40" },
  indigo: { text: "text-indigo-600 dark:text-indigo-400", bar: "bg-indigo-500" },
  amber: { text: "text-amber-600 dark:text-amber-400", bar: "bg-amber-500" },
  teal: { text: "text-teal-600 dark:text-teal-400", bar: "bg-teal-500" },
  rose: { text: "text-rose-600 dark:text-rose-400", bar: "bg-rose-500" },
  violet: { text: "text-violet-600 dark:text-violet-400", bar: "bg-violet-500" },
  emerald: { text: "text-emerald-600 dark:text-emerald-400", bar: "bg-emerald-500" },
  sky: { text: "text-sky-600 dark:text-sky-400", bar: "bg-sky-500" },
  stone: { text: "text-stone-600 dark:text-stone-400", bar: "bg-stone-500" },
} as const;

export type Accent = keyof typeof ACCENTS;

interface SectionHeadingProps {
  eyebrow: string;
  title: string;
  description?: string;
  className?: string;
  accent?: Accent;
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
  accent = "default",
}: SectionHeadingProps) {
  const accentStyle = ACCENTS[accent];
  return (
    <div className={cn("mb-8 max-w-3xl", className)}>
      <p
        className={cn(
          "flex items-center gap-3 text-xs font-medium uppercase tracking-[0.16em]",
          accentStyle.text
        )}
      >
        <span
          aria-hidden
          className={cn("inline-block h-3 w-[3px] rounded-full", accentStyle.bar)}
        />
        <span>{eyebrow}</span>
      </p>
      <h2 className="font-heading mt-3 text-3xl font-medium tracking-tight text-foreground sm:text-[2rem]">
        {title}
      </h2>
      {description ? (
        <p className="mt-4 text-base leading-relaxed text-muted-foreground">
          {description}
        </p>
      ) : null}
    </div>
  );
}
