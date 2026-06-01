import { ImageResponse } from "next/og";

import { getHeadline } from "@/lib/data";

export const alt =
  "AI Labor Market Impact Observatory — a sigmoid fit on 8 SWE-bench data points placed Claude Mythos Preview inside its 95 % CI.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  const headline = await getHeadline();
  const mythosActual = headline.swe_bench_mythos.score.toFixed(1);
  const inflection = headline.fit_inflection_year.toFixed(2);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          backgroundColor: "#0f172a",
          color: "#f8fafc",
          padding: "64px 72px",
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

        <div style={{ display: "flex", flexDirection: "column", gap: "40px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: "56px",
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: "12px",
              }}
            >
              <div
                style={{
                  fontSize: "20px",
                  letterSpacing: "0.18em",
                  textTransform: "uppercase",
                  color: "#94a3b8",
                }}
              >
                Predicted
              </div>
              <div
                style={{
                  fontSize: "152px",
                  fontWeight: 500,
                  letterSpacing: "-0.03em",
                  lineHeight: 1,
                  color: "#94a3b8",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                82.4 %
              </div>
            </div>
            <div
              style={{
                fontSize: "100px",
                lineHeight: 1,
                color: "#475569",
                paddingBottom: "32px",
              }}
            >
              →
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: "12px",
              }}
            >
              <div
                style={{
                  fontSize: "20px",
                  letterSpacing: "0.18em",
                  textTransform: "uppercase",
                  color: "#f8fafc",
                }}
              >
                Actual
              </div>
              <div
                style={{
                  fontSize: "152px",
                  fontWeight: 500,
                  letterSpacing: "-0.03em",
                  lineHeight: 1,
                  color: "#f8fafc",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {`${mythosActual} %`}
              </div>
            </div>
          </div>

          <div
            style={{
              display: "flex",
              fontSize: "30px",
              lineHeight: 1.35,
              color: "#cbd5e1",
              maxWidth: "1020px",
              letterSpacing: "-0.005em",
            }}
          >
            {`A sigmoid fit on 8 SWE-bench data points placed Claude Mythos Preview inside its 95 % CI. Inflection year ${inflection}.`}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            gap: "32px",
            fontSize: "20px",
            color: "#94a3b8",
          }}
        >
          <span>Dragos Calin · ML Engineer</span>
          <span>github.com/dragoscalin33/ai-labor-impact</span>
        </div>
      </div>
    ),
    { ...size }
  );
}
