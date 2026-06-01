"use client";

import { useSyncExternalStore, type ReactNode } from "react";

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

function useIsMounted(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

interface Props {
  className?: string;
  children: ReactNode;
  role?: string;
  ariaLabel?: string;
  ariaDescribedBy?: string;
}

export function ChartMount({
  className,
  children,
  role,
  ariaLabel,
  ariaDescribedBy,
}: Props) {
  const mounted = useIsMounted();
  return (
    <div
      className={className}
      role={role}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
    >
      {mounted ? children : null}
    </div>
  );
}
