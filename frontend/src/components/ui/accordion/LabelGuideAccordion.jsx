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

// Example usage component
const LabelGuideAccordion = () => {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="food-labels">
        <AccordionTrigger className="text-lg font-semibold">Gıda Etiketleri Okuma Rehberi</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium">İçindekiler Listesi</h3>
              <p>İçindekiler, ağırlık olarak çoktan aza doğru sıralanır. Listenin başında yer alan maddeler, üründe en yüksek miktarda bulunanlardır.</p>
            </div>
            
            <div>
              <h3 className="font-medium">Besin Değerleri Tablosu</h3>
              <p>100g/ml başına ve porsiyon başına değerler karşılaştırılmalıdır. Günlük Referans Alım Değeri (%) sütunu, bir porsiyonun günlük ihtiyacınızın ne kadarını karşıladığını gösterir.</p>
              <ul className="list-disc pl-5 mt-2">
                <li>Enerji (kalori): kJ ve kcal cinsinden belirtilir</li>
                <li>Yağlar: Doymuş, tekli doymamış ve çoklu doymamış yağ içeriği</li>
                <li>Karbonhidratlar: Şekerler dahil</li>
                <li>Protein</li>
                <li>Tuz</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">E-Kodları</h3>
              <p>E-kodları, Avrupa Birliği tarafından onaylanmış gıda katkı maddelerini belirtir:</p>
              <ul className="list-disc pl-5 mt-2">
                <li>E100-E199: Renklendiriciler</li>
                <li>E200-E299: Koruyucular</li>
                <li>E300-E399: Antioksidanlar, asitlik düzenleyiciler</li>
                <li>E400-E499: Kıvam arttırıcılar, stabilizatörler</li>
                <li>E500-E599: Asitlik düzenleyiciler, topaklanmayı önleyiciler</li>
                <li>E600-E699: Tatlandırıcılar</li>
                <li>E900-E999: Diğer (köpük önleyiciler, parlatıcılar vb.)</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">Alerjenler</h3>
              <p>Yaygın alerjenler (gluten, süt, yumurta, fıstık, kabuklu yemişler, soya vb.) etikette kalın harflerle veya farklı renkte belirtilmelidir.</p>
            </div>
            
            <div>
              <h3 className="font-medium">Son Kullanma Tarihi</h3>
              <p>"Son kullanma tarihi" geçtikten sonra ürün tüketilmemelidir. "Tavsiye edilen tüketim tarihi" (TETT) ise ürünün en iyi kalitede olduğu tarihi belirtir, bu tarihten sonra kalite düşebilir ancak güvenle tüketilebilir.</p>
            </div>
            
            <div>
              <h3 className="font-medium">Saklama Koşulları</h3>
              <p>Ürünün nasıl saklanması gerektiğine dair talimatlar (örn. "Serin ve kuru yerde saklayınız", "Açıldıktan sonra buzdolabında saklayınız").</p>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
      
      <AccordionItem value="cosmetic-labels">
        <AccordionTrigger className="text-lg font-semibold">Kozmetik Etiketleri Okuma Rehberi</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium">İçerik Listesi (INCI)</h3>
              <p>Kozmetik ürünlerde içerikler, Uluslararası Kozmetik İçerik Adlandırması (INCI) sistemine göre listelenir. %1'den fazla olan içerikler çoktan aza doğru, %1'den az olanlar herhangi bir sırada listelenebilir.</p>
            </div>
            
            <div>
              <h3 className="font-medium">Semboller ve Anlamları</h3>
              <ul className="list-disc pl-5 mt-2">
                <li>PAO (Period After Opening): Açıldıktan sonra kullanılabileceği süre (örn. 12M = 12 ay)</li>
                <li>Saklama süresi (kum saati sembolü): Minimum raf ömrü</li>
                <li>e-işareti: Ürünün AB standartlarına uygun olarak doldurulduğunu gösterir</li>
                <li>Geri dönüşüm sembolleri: Ambalajın geri dönüştürülebilir olup olmadığını belirtir</li>
                <li>Tavşan sembolü: Hayvanlar üzerinde test edilmediğini gösterir</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">Potansiyel Zararlı İçerikler</h3>
              <p>Dikkat edilmesi gereken bazı içerikler:</p>
              <ul className="list-disc pl-5 mt-2">
                <li>Parabenler (Methylparaben, Propylparaben): Koruyucu maddeler, endokrin sistem üzerinde potansiyel etkileri vardır</li>
                <li>Sentetik parfümler (Fragrance/Parfum): Cilt hassasiyeti ve alerjik reaksiyonlara neden olabilir</li>
                <li>Sülfatlar (Sodium Lauryl Sulfate, SLS): Cildi tahriş edebilir</li>
                <li>Ftalatlar: Endokrin bozucu etkiye sahip olabilir</li>
                <li>Triklosan: Antibakteriyel madde, hormon dengesini etkileyebilir</li>
                <li>Mineral yağlar (Mineral Oil, Paraffinum Liquidum): Gözenekleri tıkayabilir</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">İddialar ve Etiketleme</h3>
              <p>Kozmetik ürünlerdeki yaygın iddialar ve anlamları:</p>
              <ul className="list-disc pl-5 mt-2">
                <li>Hipoalerjenik: Alerjik reaksiyon riski düşük (ancak sıfır değil)</li>
                <li>Dermatolojik olarak test edilmiştir: Deri üzerinde test edilmiş, ancak alerji yapmayacağı garanti edilmez</li>
                <li>Doğal/Organik: İçeriklerin doğal veya organik olduğunu belirtir, ancak tüm ürün doğal olmayabilir</li>
                <li>Paraben içermez: Paraben koruyucuları içermediğini belirtir</li>
                <li>Non-comedogenic: Gözenekleri tıkamaması için formüle edilmiştir</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">Üretici Bilgileri ve Menşe</h3>
              <p>Etikette üretici firma bilgileri, iletişim adresi ve menşe ülke bilgisi yer almalıdır.</p>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
      
      <AccordionItem value="label-reading-tips">
        <AccordionTrigger className="text-lg font-semibold">Etiket Okumada Dikkat Edilmesi Gerekenler</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium">Sağlıklı Seçim İpuçları</h3>
              <ul className="list-disc pl-5 mt-2">
                <li>İçindekiler listesi ne kadar kısaysa, genellikle o kadar iyidir</li>
                <li>Tanımadığınız veya telaffuz edemediğiniz maddeleri araştırın</li>
                <li>İlk beş içeriğe özellikle dikkat edin, üründe en çok bunlar bulunur</li>
                <li>Gıdalarda, şeker içeren maddelerin farklı isimlerle listelenebileceğini unutmayın (sukroz, fruktoz, maltoz, dekstroz vb.)</li>
                <li>Benzer ürünleri karşılaştırın ve en az işlenmiş olanı seçin</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">Pazarlama Hilelerine Dikkat</h3>
              <p>Etiket ön yüzündeki iddialara kanmayın:</p>
              <ul className="list-disc pl-5 mt-2">
                <li>"Doğal" terimi yasal olarak sıkı düzenlenmez</li>
                <li>"Az yağlı" ürünlerde genellikle şeker miktarı yüksektir</li>
                <li>"Şeker ilavesiz" ürünler doğal şeker içerebilir veya yapay tatlandırıcı kullanılmış olabilir</li>
                <li>"Glutensiz" ürünler otomatik olarak daha sağlıklı değildir</li>
                <li>Kozmetikte "tüm doğal" ifadesi yanıltıcı olabilir, içindekiler listesini kontrol edin</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-medium">Sertifikalar ve Logolar</h3>
              <p>Güvenilir sertifikalar ürünlerin belli standartlara uygunluğunu gösterir:</p>
              <ul className="list-disc pl-5 mt-2">
                <li>Organik sertifikaları (ICEA, ECOCERT, vb.)</li>
                <li>Vegan/Vejetaryen logoları</li>
                <li>Helal/Kosher sertifikaları</li>
                <li>Adil Ticaret (Fair Trade) logosu</li>
                <li>Cruelty-Free (hayvanlar üzerinde test edilmemiş) sembolü</li>
              </ul>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
};

export { 
  Accordion, 
  AccordionItem, 
  AccordionTrigger, 
  AccordionContent,
  LabelGuideAccordion 
};