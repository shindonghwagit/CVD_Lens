"use client";

import { createContext, useContext, ReactNode } from "react";
import { useCVDModel, CVDType } from "../hooks/useCVDModel";

interface ModelContextValue {
  ready: boolean;
  error: string | null;
  infer: (imageData: ImageData, cvdType: CVDType) => Promise<ImageData>;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function ModelProvider({ children }: { children: ReactNode }) {
  const model = useCVDModel();
  return <ModelContext.Provider value={model}>{children}</ModelContext.Provider>;
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModel must be used inside ModelProvider");
  return ctx;
}
