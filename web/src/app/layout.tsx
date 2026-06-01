import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://ai-labor-impact.vercel.app";

const TITLE = "AI Labor Market Impact Observatory — Dragos Calin";
const DESCRIPTION =
  "A reproducible model of how AI capability progression maps onto sector-by-sector labour displacement, with vectorised Monte Carlo, hierarchical Bayesian sector model, and temporal cross-validation.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  applicationName: "AI Labor Market Impact Observatory",
  authors: [{ name: "Dragos Calin" }],
  creator: "Dragos Calin",
  keywords: [
    "AI",
    "labor market",
    "Monte Carlo",
    "Bayesian",
    "capability curve",
    "SWE-bench",
    "machine learning",
    "uncertainty propagation",
    "temporal cross-validation",
  ],
  alternates: {
    canonical: "/",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: "AI Labor Market Impact Observatory",
    title: TITLE,
    description: DESCRIPTION,
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
