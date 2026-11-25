"use client";

import { LabelGuideAccordion } from "@/components/ui/accordion/LabelGuideAccordion";

export default function LabelsPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Etiket Okuma Rehberi</h1>
      <LabelGuideAccordion />
    </div>
  );
}
