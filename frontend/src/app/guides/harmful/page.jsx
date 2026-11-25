"use client";

import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Search } from 'lucide-react';

export default function HarmfulGuide() {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  const [filteredIngredients, setFilteredIngredients] = useState([]);

  // Tüm zararlı içerikler listesi
  const harmfulIngredients = [
    {
      id: 1,
      name: "Parabenler",
      aliases: ["Metilparaben", "Propilparaben", "Bütilparaben", "Etilparaben"],
      category: ["kozmetik"],
      dangerLevel: "yuksek",
      effects: "Parabenler, kozmetik ürünlerde koruyucu olarak kullanılır. Endokrin sistemini bozabilir, hormon dengesizliğine yol açabilir ve bazı çalışmalarda meme kanseri riski ile ilişkilendirilmiştir.",
      commonProducts: ["Kremler", "Şampuanlar", "Nemlendirici", "Deodorantlar"],
      alternatives: ["Benzil alkol", "Sorbik asit", "Potasyum sorbat", "E vitaminli ürünler"],
      img: "/images/parabens.jpg"
    },
    {
      id: 2,
      name: "Sodyum Lauril Sülfat (SLS)",
      aliases: ["Sodium Lauryl Sulfate", "Sodium Laureth Sulfate", "SLES"],
      category: ["kozmetik"],
      dangerLevel: "orta",
      effects: "Güçlü bir köpürtücü olarak temizlik ve kişisel bakım ürünlerinde kullanılır. Ciltte kuruluk, tahriş ve alerji yapabilir, cildin doğal koruyucu bariyerini zayıflatabilir.",
      commonProducts: ["Şampuanlar", "Duş jelleri", "Diş macunları", "Temizlik ürünleri"],
      alternatives: ["Hindistan cevizi türevi temizleyiciler", "Decyl glucoside", "Doğal sabunlar"],
      img: "/images/sls.jpg"
    },
    {
      id: 3,
      name: "Yapay Tatlandırıcılar",
      aliases: ["Aspartam", "Sakkarin", "Sukraloz", "Asesulfam-K"],
      category: ["gida"],
      dangerLevel: "orta",
      effects: "Bazı araştırmalara göre yapay tatlandırıcılar, metabolizmayı bozabilir, iştah kontrolünü etkileyebilir ve bağırsak mikrobiyotasına zarar verebilir.",
      commonProducts: ["Diyet içecekler", "Light ürünler", "Sakızlar", "Şekersiz gıdalar"],
      alternatives: ["Stevia", "Akçaağaç şurubu", "Bal", "Hindistan cevizi şekeri"],
      img: "/images/sweeteners.jpg"
    },
    {
      id: 4,
      name: "Trans Yağlar",
      aliases: ["Kısmen hidrojenize yağlar", "Hidrojenize bitkisel yağ"],
      category: ["gida"],
      dangerLevel: "yuksek",
      effects: "Kalp hastalığı riskini artırır, HDL (iyi) kolesterolü düşürürken LDL (kötü) kolesterolü yükseltir ve kronik iltihaplanmaya neden olabilir.",
      commonProducts: ["Margarin", "Hazır kekler", "Cips", "Atıştırmalıklar", "Fast food"],
      alternatives: ["Zeytinyağı", "Hindistan cevizi yağı", "Avokado yağı", "Tereyağı"],
      img: "/images/trans-fats.jpg"
    },
    {
      id: 5,
      name: "Ftalatlar",
      aliases: ["DBP", "DEHP", "BBP", "DEP"],
      category: ["kozmetik", "temizlik"],
      dangerLevel: "yuksek",
      effects: "Hormonal dengeyi bozabilir, doğurganlığı etkileyebilir ve genel üreme sağlığı sorunlarına neden olabilir. Özellikle erkek fetüsler için zararlı olabilir.",
      commonProducts: ["Parfümler", "Ojeler", "Plastik ambalajlar", "Plastik oyuncaklar"],
      alternatives: ["Ftalat içermeyen parfümler", "Su bazlı ojeler", "Cam ambalajlı ürünler"],
      img: "/images/phthalates.jpg"
    },
    {
      id: 6,
      name: "Nitrit ve Nitratlar",
      aliases: ["Sodyum nitrit", "Potasyum nitrat", "E249", "E250", "E251", "E252"],
      category: ["gida"],
      dangerLevel: "orta",
      effects: "İşlenmiş etlerde koruyucu olarak kullanılır. Vücutta nitrozaminlere dönüşebilir, bu da kanser riskini artırabilir.",
      commonProducts: ["Salam", "Sosis", "Jambon", "Pastırma", "İşlenmiş etler"],
      alternatives: ["Taze et", "Organik işlenmiş etler", "Bitkisel protein kaynakları"],
      img: "/images/nitrates.jpg"
    },
    {
      id: 7,
      name: "Sentetik Boyalar",
      aliases: ["FD&C Blue 1", "Yellow 5", "Red 40", "Tartrazin", "E102", "E129", "E133"],
      category: ["gida", "kozmetik"],
      dangerLevel: "orta",
      effects: "Hiperaktivite, dikkat eksikliği ve alerjik reaksiyonlarla ilişkilendirilmiştir. Astım semptomlarını tetikleyebilir.",
      commonProducts: ["Şekerlemeler", "Gazlı içecekler", "Renklendirici kozmetikler", "Renkli sakızlar"],
      alternatives: ["Doğal gıda boyaları", "Pancar suyu", "Zerdeçal", "Spirulina"],
      img: "/images/artificial-colors.jpg"
    },
    {
      id: 8,
      name: "BHA ve BHT",
      aliases: ["Bütillenmiş hidroksianisol", "Bütillenmiş hidroksitoluen", "E320", "E321"],
      category: ["gida", "kozmetik"],
      dangerLevel: "orta",
      effects: "Gıda ve kozmetikte koruyucu olarak kullanılır. Hormonal dengeyi bozabilir ve uzun vadede karaciğer, tiroid ve böbrek sorunlarına yol açabilir.",
      commonProducts: ["Paketli atıştırmalıklar", "Cipsler", "Yağlar", "Ruj ve fondöten"],
      alternatives: ["E vitamini", "Rozmarin ekstresi", "Askorbik asit"],
      img: "/images/bha-bht.jpg"
    },
    {
      id: 9,
      name: "Formaldehit",
      aliases: ["Formalin", "Metanal", "Quarternium-15", "DMDM Hydantoin", "Imidazolidinyl Urea"],
      category: ["kozmetik", "temizlik"],
      dangerLevel: "yuksek",
      effects: "Güçlü bir koruyucu maddedir. Alerjik reaksiyonlara, ciltte tahrişe ve solunum sorunlarına neden olabilir. Uluslararası Kanser Araştırma Ajansı tarafından kanserojen olarak sınıflandırılmıştır.",
      commonProducts: ["Tırnak ürünleri", "Saç düzleştiriciler", "Saç boyaları", "Bazı şampuanlar"],
      alternatives: ["Doğal koruyuculu ürünler", "Formaldehit içermeyen tırnak ürünleri"],
      img: "/images/formaldehyde.jpg"
    },
    {
      id: 10,
      name: "Yüksek Fruktozlu Mısır Şurubu",
      aliases: ["HFCS", "Corn sugar", "Glucose-fructose syrup"],
      category: ["gida"],
      dangerLevel: "orta",
      effects: "İnsülin direnci, obezite, yüksek kolesterol ve kalp hastalıkları ile ilişkilendirilmiştir. Metabolik sendrom riskini artırabilir.",
      commonProducts: ["Gazlı içecekler", "Paketli tatlılar", "Meyve suları", "İşlenmiş gıdalar"],
      alternatives: ["Akçaağaç şurubu", "Bal", "Hindistan cevizi şekeri", "Stevia"],
      img: "/images/hfcs.jpg"
    },
    {
      id: 11,
      name: "Titanium Dioksit",
      aliases: ["E171", "CI 77891", "Titanyum beyazı"],
      category: ["gida", "kozmetik"],
      dangerLevel: "orta",
      effects: "Gıdalarda ve kozmetikte beyazlatıcı olarak kullanılır. Bazı ülkelerde gıdalarda kullanımı yasaklanmaya başlanmıştır çünkü nanopartikülleri vücutta birikebilir ve iltihaplanmaya neden olabilir.",
      commonProducts: ["Şekerlemeler", "Sakızlar", "Güneş kremleri", "Makyaj ürünleri"],
      alternatives: ["Mineral bazlı güneş kremleri", "Çinko oksit", "Organik güneş kremleri"],
      img: "/images/titanium-dioxide.jpg"
    },
    {
      id: 12,
      name: "MSG (Monosodyum Glutamat)",
      aliases: ["E621", "Vetsin", "Çin tuzu", "Ajinomoto"],
      category: ["gida"],
      dangerLevel: "orta",
      effects: "Lezzet artırıcı olarak kullanılır. Bazı insanlarda baş ağrısı, çarpıntı, terleme gibi 'Çin Lokantası Sendromu' belirtilerine neden olabilir.",
      commonProducts: ["Hazır çorbalar", "Cipsler", "Paketli et ürünleri", "Fast food"],
      alternatives: ["Doğal baharatlar", "Tuz", "Maya ekstraktı"],
      img: "/images/msg.jpg"
    }
  ];

  // Tehlike işaretleri ve anlamları
  const dangerSymbols = [
    {
      id: 1,
      name: "Patlayıcı",
      symbol: "💥",
      description: "Patlayıcı özelliği olan maddeler. Basınç, ısı veya şok etkisiyle patlayabilirler.",
      img: "/images/explosive.png"
    },
    {
      id: 2,
      name: "Yanıcı",
      symbol: "🔥",
      description: "Kolay tutuşabilen maddeler. Alev, kıvılcım veya ısı ile kolayca yanabilirler.",
      img: "/images/flammable.png"
    },
    {
      id: 3,
      name: "Oksitleyici",
      symbol: "⭕",
      description: "Oksidasyon reaksiyonunu destekleyen maddeler. Yangınları şiddetlendirebilirler.",
      img: "/images/oxidizing.png"
    },
    {
      id: 4,
      name: "Aşındırıcı",
      symbol: "🧪",
      description: "Cilde, gözlere ve solunum sistemine zarar verebilen asitler veya bazlar.",
      img: "/images/corrosive.png"
    },
    {
      id: 5,
      name: "Toksik",
      symbol: "☠️",
      description: "Az miktarda bile sağlığa ciddi zarar verebilen veya ölümcül olabilen maddeler.",
      img: "/images/toxic.png"
    },
    {
      id: 6,
      name: "Sağlık Tehlikesi",
      symbol: "⚕️",
      description: "Kronik sağlık sorunlarına, kansere, doğumsal anomalilere veya genetik mutasyonlara neden olabilecek maddeler.",
      img: "/images/health-hazard.png"
    },
    {
      id: 7,
      name: "Çevresel Tehlike",
      symbol: "🌍",
      description: "Su yaşamı ve ekosistem için zararlı olan, çevrede uzun süreli olumsuz etkilere neden olabilen maddeler.",
      img: "/images/environmental-hazard.png"
    },
    {
      id: 8,
      name: "Basınçlı Gaz",
      symbol: "🧴",
      description: "Basınç altında sıkıştırılmış gazlar. Isındığında patlayabilir.",
      img: "/images/gas-cylinder.png"
    }
  ];

  // Arama ve filtreleme fonksiyonu
  useEffect(() => {
    const filtered = harmfulIngredients.filter(ingredient => {
      // Kategori filtrelemesi
      const categoryMatch = activeTab === "all" || ingredient.category.includes(activeTab);
      
      // Arama filtrelemesi
      const searchMatch = 
        ingredient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ingredient.aliases.some(alias => alias.toLowerCase().includes(searchTerm.toLowerCase())) ||
        ingredient.commonProducts.some(product => product.toLowerCase().includes(searchTerm.toLowerCase()));
      
      return categoryMatch && searchMatch;
    });
    
    setFilteredIngredients(filtered);
  }, [searchTerm, activeTab]);

  // Tehlike seviyesi rengini döndüren yardımcı fonksiyon
  const getDangerColor = (level) => {
    switch(level) {
      case "yuksek": return "text-red-600 bg-red-100 dark:bg-red-900 dark:bg-opacity-30";
      case "orta": return "text-orange-600 bg-orange-100 dark:bg-orange-900 dark:bg-opacity-30";
      case "dusuk": return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:bg-opacity-30";
      default: return "text-gray-600 bg-gray-100";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 dark:text-gray-100 transition-colors duration-200">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-12 px-4 shadow-lg">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Zararlı İçerikler Rehberi</h1>
          <p className="mt-4 text-xl max-w-3xl mx-auto">
            Bilimsel araştırmalar ışığında gıda, kozmetik ve temizlik ürünlerindeki zararlı içerikleri tanıyın, sağlıklı alternatifler keşfedin.
          </p>
          
          {/* Arama kutusu */}
          <div className="mt-8 max-w-2xl mx-auto relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-4 py-3 border border-transparent rounded-full focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-900"
              placeholder="İçerik ismi veya ürün ara..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </header>

      {/* İçerik */}
      <main className="max-w-6xl mx-auto py-10 px-4">
        {/* Tanıtım ve Önem */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-3xl font-semibold text-blue-600 dark:text-blue-400 mb-4">Ürün İçeriklerini Tanımak Neden Önemli?</h2>
              <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed mb-4">
                Her gün kullandığımız ürünlerdeki içerikler, zaman içinde vücudumuzda birikebilir ve çeşitli sağlık sorunlarına yol açabilir. Özellikle bazı katkı maddeleri, koruyucular ve sentetik kimyasallar, 
                endokrin sistemimizi bozabilir, alerjik reaksiyonlara neden olabilir veya daha ciddi sağlık risklerini artırabilir.
              </p>
              <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed">
                Bu rehber, en yaygın zararlı içerikleri, bu içeriklerin bulunduğu ürünleri, potansiyel sağlık etkilerini ve daha güvenli alternatifleri tanımanıza yardımcı olmak için hazırlanmıştır.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg">
              <div className="aspect-video bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                <div className="text-center p-4">
                  <h3 className="text-2xl font-bold text-blue-700 dark:text-blue-300 mb-2">Biliyor muydunuz?</h3>
                  <ul className="text-gray-700 dark:text-gray-300 space-y-3">
                    <li className="flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      <span>Ortalama bir kadın, her gün 168 farklı kimyasala maruz kalıyor.</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      <span>Kozmetik ürünlerdeki içeriklerin %60'ı ciltten emilerek kana karışabilir.</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      <span>Gıda etiketlerinde "doğal aroma" ibaresi 50'den fazla farklı kimyasalı maskeleyebilir.</span>
                    </li>
                    <li className="flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      <span>İşlenmiş gıdalarda 3000'den fazla farklı katkı maddesi kullanılmaktadır.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* İçerik Kategorileri */}
        <section className="mb-16">
          <Tabs defaultValue="all" onValueChange={setActiveTab}>
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
              <h2 className="text-3xl font-semibold text-blue-600 dark:text-blue-400 mb-4 md:mb-0">Zararlı İçerikler</h2>
              <TabsList className="bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
                <TabsTrigger value="all" className="px-4 py-2 rounded-md">Tümü</TabsTrigger>
                <TabsTrigger value="kozmetik" className="px-4 py-2 rounded-md">Kozmetik</TabsTrigger>
                <TabsTrigger value="gida" className="px-4 py-2 rounded-md">Gıda</TabsTrigger>
                <TabsTrigger value="temizlik" className="px-4 py-2 rounded-md">Temizlik</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="all" className="mt-0">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredIngredients.length > 0 ? (
                  filteredIngredients.map((ingredient) => (
                    <div key={ingredient.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden flex flex-col">
                      <div className="h-48 bg-gray-200 dark:bg-gray-700 overflow-hidden">
                        {/* Gerçek resim yerine placeholder gösteriliyor */}
                        <div className="w-full h-full bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900 dark:to-purple-900 flex items-center justify-center">
                          <span className="text-4xl">{ingredient.category.includes("kozmetik") ? "🧴" : ingredient.category.includes("gida") ? "🍎" : "🧼"}</span>
                        </div>
                      </div>
                      
                      <div className="p-5 flex-grow">
                        <div className="flex justify-between items-start mb-2">
                          <h3 className="text-xl font-bold text-gray-800 dark:text-white">{ingredient.name}</h3>
                          <span className={`text-xs font-semibold px-2 py-1 rounded-full uppercase ${getDangerColor(ingredient.dangerLevel)}`}>
                            {ingredient.dangerLevel === "yuksek" ? "Yüksek Risk" : 
                             ingredient.dangerLevel === "orta" ? "Orta Risk" : "Düşük Risk"}
                          </span>
                        </div>
                        
                        <div className="mb-3">
                          <span className="text-sm text-gray-500 dark:text-gray-400">Diğer isimler:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {ingredient.aliases.map((alias, index) => (
                              <span key={index} className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">
                                {alias}
                              </span>
                            ))}
                          </div>
                        </div>
                        
                        <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 line-clamp-3">
                          {ingredient.effects}
                        </p>
                        
                        <div className="mb-3">
                          <span className="text-sm text-gray-500 dark:text-gray-400">Bulunduğu ürünler:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {ingredient.commonProducts.map((product, index) => (
                              <span key={index} className="text-xs bg-red-50 dark:bg-red-900 dark:bg-opacity-20 text-red-600 dark:text-red-300 px-2 py-1 rounded">
                                {product}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="p-5 bg-gray-50 dark:bg-gray-800 border-t dark:border-gray-700">
                        <Accordion type="single" collapsible>
                          <AccordionItem value="alternatives">
                            <AccordionTrigger className="text-sm text-blue-600 dark:text-blue-400">
                              Güvenli Alternatifler
                            </AccordionTrigger>
                            <AccordionContent>
                              <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-1 pl-5 list-disc">
                                {ingredient.alternatives.map((alt, index) => (
                                  <li key={index}>{alt}</li>
                                ))}
                              </ul>
                            </AccordionContent>
                          </AccordionItem>
                        </Accordion>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="col-span-3 text-center py-12">
                    <p className="text-xl text-gray-500 dark:text-gray-400">Arama kriterlerinize uygun içerik bulunamadı.</p>
                    <button 
                      onClick={() => {setSearchTerm(""); setActiveTab("all");}}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                    >
                      Filtreleri Temizle
                    </button>
                  </div>
                )}
              </div>
            </TabsContent>
            
            <TabsContent value="kozmetik" className="mt-0">
              {/* Kozmetik ürünler için içerik aynı düzende gösterilecek */}
            </TabsContent>
            
            <TabsContent value="gida" className="mt-0">
              {/* Gıda ürünleri için içerik aynı düzende gösterilecek */}
            </TabsContent>
            
            <TabsContent value="temizlik" className="mt-0">
              {/* Temizlik ürünleri için içerik aynı düzende gösterilecek */}
            </TabsContent>
          </Tabs>
        </section>

        {/* Tehlike İşaretleri Rehberi */}
        <section className="mb-16">
          <h2 className="text-3xl font-semibold text-blue-600 dark:text-blue-400 mb-8">Tehlike İşaretleri Rehberi</h2>
          <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed mb-8">
            Ürün etiketlerinde ve ambalajlarında görebileceğiniz bu işaretler, içeriğin potansiyel tehlikelerini belirtir. Bu işaretleri tanımak, daha bilinçli tüketim kararları almanıza yardımcı olabilir.
          </p>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {dangerSymbols.map((symbol) => (
              <div key={symbol.id} className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
                <div className="p-6 flex flex-col items-center text-center">
                  <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900 dark:bg-opacity-30 flex items-center justify-center mb-4">
                    <span className="text-3xl">{symbol.symbol}</span>
                  </div>
                  <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-2">{symbol.name}</h3>
                  <p className="text-gray-600 dark:text-gray-300 text-sm">
                    {symbol.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Sağlıklı Alışveriş Rehberi */}
        <section className="mb-16">
          <h2 className="text-3xl font-semibold text-blue-600 dark:text-blue-400 mb-6">Sağlıklı Alışveriş Rehberi</h2>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 md:p-8">
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">Ürün Seçerken</h3>
                <ul className="space-y-4">
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">İçerik Listesini İnceleyin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Ürünün içeriğinde bulunan maddeleri dikkatlice okuyun ve potansiyel riskleri araştırın.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Doğal İçeriklere Yönelin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Yapay katkı maddeleri yerine doğal ve organik içerikli ürünleri tercih edin.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Uyarı İşaretlerine Dikkat Edin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Ürün ambalajında yer alan tehlike işaretleri ve etiketlere göz atın.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Güvenilir Markaları Tercih Edin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Kalite ve güvenilirlik sertifikalarına sahip markaları seçmeye özen gösterin.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Fiyat ve Kalite Dengesini Değerlendirin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Ucuz ürünlerin her zaman kaliteli olmayabileceğini göz önünde bulundurun.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Kullanıcı Yorumlarını Okuyun</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Diğer tüketicilerin deneyimlerini inceleyerek bilinçli kararlar verin.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 text-xl mr-3 mt-0.5">✓</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Uzman Önerilerini Takip Edin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        FDA, WHO ve diğer güvenilir kuruluşların değerlendirmelerini göz önünde bulundurun.
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-4">Sağlıklı Alışveriş İpuçları</h3>
                <ul className="space-y-4">
                  <li className="flex items-start">
                    <span className="text-blue-500 text-xl mr-3 mt-0.5">ℹ️</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Etiket Okuma Teknikleri</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Ürünün içeriğinde yer alan tüm bileşenleri ve uyarı simgelerini dikkatlice okuyun.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 text-xl mr-3 mt-0.5">ℹ️</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Ürün Karşılaştırması Yapın</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Farklı markaların ürün içeriklerini karşılaştırarak en sağlıklı seçeneği belirleyin.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 text-xl mr-3 mt-0.5">ℹ️</span>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white">Organik ve Doğal Sertifikalı Ürünleri Tercih Edin</h4>
                      <p className="text-gray-600 dark:text-gray-300 text-sm">
                        Sertifikalı organik ürünler, içeriklerinin daha doğal olduğunu gösterir.
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 py-6 px-4 text-center">
        <p className="text-sm">
          &copy; {new Date().getFullYear()} Sağlıklı Ürün Rehberi. Tüm hakları saklıdır.
        </p>
      </footer>
    </div>
  );
}