"use client";

import { Tabs } from "@/components/ui/tabs";
import { useSelectedScenario } from "@/components/site/scenario-context";
import type { ScenarioKey } from "@/lib/types";

interface ScenariosTabsProps {
  className?: string;
  children: React.ReactNode;
}

export function ScenariosTabs({ className, children }: ScenariosTabsProps) {
  const { selected, setSelected } = useSelectedScenario();
  return (
    <Tabs
      value={selected}
      onValueChange={(v) => setSelected(v as ScenarioKey)}
      className={className}
    >
      {children}
    </Tabs>
  );
}
