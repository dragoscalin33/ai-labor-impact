import { CapabilityCurveSection } from "@/components/site/capability-curve-section";
import { HeroSection } from "@/components/site/hero-section";
import { InsightsSection } from "@/components/site/insights-section";
import { KeyFindingsSection } from "@/components/site/key-findings-section";
import { MethodologySection } from "@/components/site/methodology-section";
import { SelectedScenarioProvider } from "@/components/site/scenario-context";
import { ScenariosSection } from "@/components/site/scenarios-section";
import { SectorHeatmapSection } from "@/components/site/sector-heatmap-section";
import { SiteFooter } from "@/components/site/site-footer";
import { SiteHeader } from "@/components/site/site-header";
import { StackSection } from "@/components/site/stack-section";
import { ValidationSection } from "@/components/site/validation-section";

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main id="main-content" className="flex flex-1 flex-col">
        <HeroSection />
        <KeyFindingsSection />
        <CapabilityCurveSection />
        <SelectedScenarioProvider initial="base">
          <ScenariosSection />
          <SectorHeatmapSection />
          <ValidationSection />
          <InsightsSection />
        </SelectedScenarioProvider>
        <MethodologySection />
        <StackSection />
      </main>
      <SiteFooter />
    </>
  );
}
