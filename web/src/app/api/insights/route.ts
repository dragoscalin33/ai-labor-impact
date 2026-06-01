import { createGroq } from "@ai-sdk/groq";
import { streamText } from "ai";

import { getHeadline, getScenarios } from "@/lib/data";
import type { ScenarioKey } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;

const VALID_KEYS: ScenarioKey[] = [
  "optimistic",
  "base",
  "pessimistic",
  "mythos_accelerated",
];

const SYSTEM_PROMPT =
  "You are summarising a Monte Carlo unemployment scenario for a policy reader. " +
  "Stick to the supplied numbers. 4-5 sentences, sober tone, no marketing language. " +
  "Do not invent new figures, do not speculate beyond the inputs, and do not use emojis or headings.";

interface InsightsRequestBody {
  scenario?: unknown;
}

function isScenarioKey(value: unknown): value is ScenarioKey {
  return typeof value === "string" && (VALID_KEYS as string[]).includes(value);
}

export async function GET(): Promise<Response> {
  if (!process.env.GROQ_API_KEY) {
    return Response.json({ reason: "no_key" }, { status: 501 });
  }
  return Response.json(
    { reason: "method_not_allowed", hint: "POST with { scenario }" },
    { status: 405 },
  );
}

export async function POST(req: Request): Promise<Response> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return Response.json({ reason: "no_key" }, { status: 501 });
  }

  let body: InsightsRequestBody;
  try {
    body = (await req.json()) as InsightsRequestBody;
  } catch {
    return Response.json({ reason: "bad_request" }, { status: 400 });
  }

  if (!isScenarioKey(body.scenario)) {
    return Response.json({ reason: "invalid_scenario" }, { status: 400 });
  }
  const scenarioKey: ScenarioKey = body.scenario;

  const [scenarios, headline] = await Promise.all([
    getScenarios(),
    getHeadline(),
  ]);
  const series = scenarios.scenarios.find((s) => s.key === scenarioKey);
  const headEntry = headline.displacement_2035_pct.by_scenario.find(
    (s) => s.key === scenarioKey,
  );
  if (!series || !headEntry) {
    return Response.json({ reason: "scenario_not_found" }, { status: 404 });
  }

  const facts = [
    `Scenario: ${series.name} (key=${series.key}).`,
    `Policy framing: ${series.policy_lever}.`,
    `Monte Carlo samples: ${scenarios.n_samples}, seed ${scenarios.seed}.`,
    `Year range: ${scenarios.years[0]}–${scenarios.years[scenarios.years.length - 1]}.`,
    `2035 displacement (% of labour force): median ${headEntry.p50_2035.toFixed(1)}, P05 ${headEntry.p05_2035.toFixed(1)}, P95 ${headEntry.p95_2035.toFixed(1)}.`,
    `Peak displacement: median ${headEntry.peak_p50.toFixed(1)} % in ${headEntry.peak_year}.`,
    `Capability fit: SWE-bench sigmoid inflection ${headline.fit_inflection_year.toFixed(2)}, R^2 ${headline.fit_r_squared.toFixed(3)}.`,
    `Mythos checkpoint: ${headline.swe_bench_mythos.model} scored ${headline.swe_bench_mythos.score} on SWE-bench Verified (${headline.swe_bench_mythos.year.toFixed(2)}).`,
  ].join("\n");

  const groq = createGroq({ apiKey });
  const result = streamText({
    model: groq("llama-3.3-70b-versatile"),
    system: SYSTEM_PROMPT,
    prompt:
      `Write a 4-5 sentence interpretation of the following Monte Carlo unemployment scenario for a policy reader. ` +
      `Use only the supplied numbers. Plain prose, no bullets, no headings, no emojis.\n\n${facts}`,
    temperature: 0.4,
  });

  return result.toTextStreamResponse();
}
