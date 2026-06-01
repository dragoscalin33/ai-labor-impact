import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0f172a",
          color: "#f8fafc",
          fontFamily: "system-ui, sans-serif",
          fontSize: 16,
          fontWeight: 600,
          letterSpacing: "-0.02em",
          borderRadius: 6,
        }}
      >
        AL
      </div>
    ),
    { ...size }
  );
}
