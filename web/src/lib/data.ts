import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  BenchmarksData,
  HeadlineData,
  InsightsData,
  ScenariosData,
  SectorImpactData,
  ValidationData,
} from "@/lib/types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(filename: string): Promise<T> {
  const buf = await readFile(path.join(DATA_DIR, filename), "utf8");
  return JSON.parse(buf) as T;
}

export const getHeadline = (): Promise<HeadlineData> =>
  readJson<HeadlineData>("headline.json");

export const getBenchmarks = (): Promise<BenchmarksData> =>
  readJson<BenchmarksData>("benchmarks.json");

export const getScenarios = (): Promise<ScenariosData> =>
  readJson<ScenariosData>("scenarios.json");

export const getSectorImpact = (): Promise<SectorImpactData> =>
  readJson<SectorImpactData>("sector_impact.json");

export const getValidation = (): Promise<ValidationData> =>
  readJson<ValidationData>("validation.json");

export const getInsights = (): Promise<InsightsData> =>
  readJson<InsightsData>("insights.json");

export const GITHUB_REPO_URL = "https://github.com/dragoscalin33/ai-labor-impact";
export const GITHUB_BLOB = `${GITHUB_REPO_URL}/blob/main`;
