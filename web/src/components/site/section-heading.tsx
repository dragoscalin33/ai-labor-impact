import { cn } from "@/lib/utils";

interface SectionHeadingProps {
  eyebrow: string;
  title: string;
  description?: string;
  className?: string;
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
}: SectionHeadingProps) {
  return (
    <div className={cn("mb-12 max-w-3xl", className)}>
      <p className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
        {eyebrow}
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
