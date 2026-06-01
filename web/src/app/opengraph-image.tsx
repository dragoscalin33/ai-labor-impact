import { ImageResponse } from "next/og";

import { getHeadline } from "@/lib/data";

export const alt =
  "AI Labor Market Impact Observatory — reproducible model of AI capability progression mapped onto labour displacement.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  const headline = await getHeadline();
  const base = headline.displacement_2035_pct.by_scenario.find(
    (s) => s.key === "base"
  );
  const baseRange = base
    ? `${base.p05_2035.toFixed(1)}–${base.p95_2035.toFixed(1)} %`
    : "—";
  const peakLine = base ? `Peak ${base.peak_p50.toFixed(1)} % at ${base.peak_year}` : "";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "#0f172a",
          color: "#f8fafc",
          padding: "72px",
          fontFamily: "Inter, system-ui, sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            fontSize: "20px",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "#94a3b8",
          }}
        >
          <span
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "44px",
              height: "44px",
              borderRadius: "10px",
              backgroundColor: "#f8fafc",
              color: "#0f172a",
              fontSize: "20px",
              fontWeight: 600,
              letterSpacing: 0,
            }}
          >
            AL
          </span>
          <span>AI Labor Market Impact Observatory</span>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginTop: "56px",
            fontSize: "56px",
            lineHeight: 1.1,
            fontWeight: 500,
            letterSpacing: "-0.02em",
            maxWidth: "1000px",
          }}
        >
          A reproducible model of AI capability progression mapped onto labour
          displacement.
        </div>

        <div
          style={{
            display: "flex",
            marginTop: "auto",
            justifyContent: "space-between",
            alignItems: "flex-end",
            gap: "32px",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <div
              style={{
                fontSize: "16px",
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: "#94a3b8",
              }}
            >
              Base scenario · 2035 displacement (95 % MC)
            </div>
            <div
              style={{
                fontSize: "72px",
                fontWeight: 500,
                letterSpacing: "-0.02em",
                color: "#f8fafc",
              }}
            >
              {baseRange}
            </div>
            <div style={{ fontSize: "20px", color: "#cbd5e1" }}>{peakLine}</div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
              gap: "6px",
              fontSize: "18px",
              color: "#94a3b8",
            }}
          >
            <div style={{ color: "#f8fafc", fontWeight: 500 }}>
              Dragos Calin · ML Engineer
            </div>
            <div>github.com/dragoscalin33/ai-labor-impact</div>
          </div>
        </div>
      </div>
    ),
    {
      ...size,
    }
  );
}
