"use client";

import * as React from "react";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const Accordion = AccordionPrimitive.Root;

const AccordionItem = React.forwardRef(({ className, ...props }, ref) => (
  <AccordionPrimitive.Item
    ref={ref}
    className={cn("border-b", className)}
    {...props}
  />
));
AccordionItem.displayName = "AccordionItem";

const AccordionTrigger = React.forwardRef(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Header className="flex">
    <AccordionPrimitive.Trigger
      ref={ref}
      className={cn(
        "flex flex-1 items-center justify-between py-4 font-medium transition-all hover:underline [&[data-state=open]>svg]:rotate-180",
        className
      )}
      {...props}
    >
      {children}
      <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
));
AccordionTrigger.displayName = "AccordionTrigger";

const AccordionContent = React.forwardRef(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Content
    ref={ref}
    className={cn(
      "overflow-hidden text-sm transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down",
      className
    )}
    {...props}
  >
    <div className="pb-4 pt-0">{children}</div>
  </AccordionPrimitive.Content>
));
AccordionContent.displayName = "AccordionContent";

// LabelGuideAccordion bileşeni
const LabelGuideAccordion = () => {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="food-labels">
        <AccordionTrigger className="text-lg font-semibold">
          Gıda Etiketleri Okuma Rehberi
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium">İçindekiler Listesi</h3>
              <p>
                İçindekiler, ağırlık olarak çoktan aza doğru sıralanır. Listenin başında yer alan maddeler, üründe en yüksek miktarda bulunanlardır.
              </p>
            </div>
            <div>
              <h3 className="font-medium">Besin Değerleri Tablosu</h3>
              <p>
                100g/ml başına ve porsiyon başına değerler karşılaştırılmalıdır. Günlük Referans Alım Değeri (%) sütunu, bir porsiyonun günlük ihtiyacınızın ne kadarını karşıladığını gösterir.
              </p>
              <ul className="list-disc pl-5 mt-2">
                <li>Enerji (kalori): kJ ve kcal cinsinden belirtilir</li>
                <li>Yağlar: Doymuş, tekli doymamış ve çoklu doymamış yağ içeriği</li>
                <li>Karbonhidratlar: Şekerler dahil</li>
                <li>Protein</li>
                <li>Tuz</li>
              </ul>
            </div>
            {/* Diğer içerikler */}
          </div>
        </AccordionContent>
      </AccordionItem>
      {/* Diğer AccordionItem bileşenleri */}
    </Accordion>
  );
};

export {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
  LabelGuideAccordion,
};
