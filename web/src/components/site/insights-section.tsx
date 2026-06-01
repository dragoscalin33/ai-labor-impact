import { AiInsights } from "@/components/site/ai-insights";
import { SectionHeading } from "@/components/site/section-heading";
import { getInsights } from "@/lib/data";

export async function InsightsSection() {
  const insights = await getInsights();

  return (
    <section
      id="insights"
      className="border-t border-border/60 px-6 py-16"
    >
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="rose"
          eyebrow="AI insights"
          title="Per-scenario narrative — curated by default, streamable on demand."
          description="Reads the cached narrative for whichever scenario tab is active above. Switch to Live (LLM) to stream a fresh 4-5 sentence interpretation through the Vercel AI SDK and Groq (llama-3.3-70b-versatile). Without a GROQ_API_KEY, the endpoint returns 501 and the panel falls back to the cached text automatically."
        />
        <AiInsights insights={insights} />
      </div>
    </section>
  );
}
