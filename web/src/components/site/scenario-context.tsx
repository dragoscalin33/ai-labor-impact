"use client";

import { createContext, useContext, useMemo, useState } from "react";

import type { ScenarioKey } from "@/lib/types";

interface SelectedScenarioContextValue {
  selected: ScenarioKey;
  setSelected: (key: ScenarioKey) => void;
}

const SelectedScenarioContext =
  createContext<SelectedScenarioContextValue | null>(null);

interface SelectedScenarioProviderProps {
  initial?: ScenarioKey;
  children: React.ReactNode;
}

export function SelectedScenarioProvider({
  initial = "base",
  children,
}: SelectedScenarioProviderProps) {
  const [selected, setSelected] = useState<ScenarioKey>(initial);
  const value = useMemo(() => ({ selected, setSelected }), [selected]);
  return (
    <SelectedScenarioContext.Provider value={value}>
      {children}
    </SelectedScenarioContext.Provider>
  );
}

export function useSelectedScenario(): SelectedScenarioContextValue {
  const ctx = useContext(SelectedScenarioContext);
  if (!ctx) {
    throw new Error(
      "useSelectedScenario must be used inside <SelectedScenarioProvider>",
    );
  }
  return ctx;
}
