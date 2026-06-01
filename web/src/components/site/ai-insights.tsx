"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { useSelectedScenario } from "@/components/site/scenario-context";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type {
  InsightsData,
  InsightsScenarioEntry,
  ScenarioKey,
} from "@/lib/types";

type Mode = "curated" | "live";

const SHORT_LABEL: Record<ScenarioKey, string> = {
  optimistic: "Optimistic",
  base: "Base",
  pessimistic: "Pessimistic",
  mythos_accelerated: "Mythos",
};

interface AiInsightsProps {
  insights: InsightsData;
}

interface LiveState {
  text: string;
  status: "idle" | "streaming" | "done" | "error" | "no_key";
  error?: string;
}

const INITIAL_LIVE: LiveState = { text: "", status: "idle" };

export function AiInsights({ insights }: AiInsightsProps) {
  const { selected } = useSelectedScenario();
  const [mode, setMode] = useState<Mode>("curated");
  const [liveByKey, setLiveByKey] = useState<Record<ScenarioKey, LiveState>>({
    optimistic: INITIAL_LIVE,
    base: INITIAL_LIVE,
    pessimistic: INITIAL_LIVE,
    mythos_accelerated: INITIAL_LIVE,
  });

  const abortRef = useRef<AbortController | null>(null);

  const cached: InsightsScenarioEntry = insights.scenarios[selected];
  const liveState = liveByKey[selected];

  const noticeNoKey = liveState.status === "no_key";

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  function patchLive(key: ScenarioKey, patch: Partial<LiveState>): void {
    setLiveByKey((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));
  }

  async function runLive(key: ScenarioKey): Promise<void> {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    patchLive(key, { text: "", status: "streaming", error: undefined });

    let res: Response;
    try {
      res = await fetch("/api/insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: key }),
        signal: ctrl.signal,
      });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      patchLive(key, {
        status: "error",
        error: err instanceof Error ? err.message : "network error",
      });
      return;
    }

    if (res.status === 501) {
      patchLive(key, { status: "no_key" });
      return;
    }
    if (!res.ok || !res.body) {
      patchLive(key, {
        status: "error",
        error: `request failed (${res.status})`,
      });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let acc = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        patchLive(key, { text: acc });
      }
      acc += decoder.decode();
      patchLive(key, { text: acc, status: "done" });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      patchLive(key, {
        status: "error",
        error: err instanceof Error ? err.message : "stream error",
      });
    }
  }

  function handleSelectLive(): void {
    setMode("live");
    if (liveState.status === "idle" || liveState.status === "error") {
      void runLive(selected);
    }
  }

  function handleSelectCurated(): void {
    setMode("curated");
    abortRef.current?.abort();
  }

  function handleRegenerate(): void {
    void runLive(selected);
  }

  useEffect(() => {
    if (mode !== "live") return;
    const s = liveByKey[selected];
    if (s.status !== "idle" && s.status !== "error") return;
    const handle = setTimeout(() => {
      void runLive(selected);
    }, 0);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, mode]);

  const showCurated = mode === "curated" || noticeNoKey;

  const bodyParagraphs = useMemo(() => {
    if (showCurated) {
      return cached.narrative.split(/\n+/).filter(Boolean);
    }
    return liveState.text ? [liveState.text] : [];
  }, [showCurated, cached.narrative, liveState.text]);

  return (
    <div className="rounded-xl border border-border/70 bg-card">
      <div className="flex flex-col gap-3 border-b border-border/60 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="uppercase tracking-[0.12em]">
            {SHORT_LABEL[selected]}
          </Badge>
          <p className="text-sm font-medium text-foreground">
            {cached.headline}
          </p>
        </div>
        <div
          role="tablist"
          aria-label="Commentary mode"
          className="inline-flex items-center gap-1 self-start rounded-lg border border-border/70 p-1 sm:self-auto"
        >
          <Button
            type="button"
            role="tab"
            aria-selected={mode === "curated"}
            size="sm"
            variant={mode === "curated" ? "secondary" : "ghost"}
            onClick={handleSelectCurated}
          >
            Curated
          </Button>
          <Button
            type="button"
            role="tab"
            aria-selected={mode === "live"}
            size="sm"
            variant={mode === "live" ? "secondary" : "ghost"}
            onClick={handleSelectLive}
            disabled={liveState.status === "streaming"}
          >
            {liveState.status === "streaming" ? "Streaming…" : "Live (LLM)"}
          </Button>
        </div>
      </div>

      <div className="px-6 py-6">
        {mode === "live" && noticeNoKey ? (
          <div className="mb-4 rounded-md border border-border/70 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            Live commentary requires <code>GROQ_API_KEY</code> — showing curated
            narrative.
          </div>
        ) : null}

        {mode === "live" &&
        liveState.status === "error" &&
        liveState.error ? (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            Live request failed: {liveState.error}. Showing curated narrative
            below.
          </div>
        ) : null}

        <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
          {bodyParagraphs.length === 0 ? (
            <p className="italic">Generating live commentary…</p>
          ) : (
            bodyParagraphs.map((para, i) => <p key={i}>{para}</p>)
          )}
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-4 text-xs text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Policy lever.</span>{" "}
            {cached.policy_lever}
          </p>
          {mode === "live" && !noticeNoKey ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={handleRegenerate}
              disabled={liveState.status === "streaming"}
            >
              {liveState.status === "streaming"
                ? "Streaming…"
                : "Regenerate"}
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
