import React from "react";
import { LabelGuideAccordion } from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertTriangle, Leaf, ShoppingBag, Heart } from "lucide-react";

export default function AlternativesGuide() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-100 to-gray-200">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-700 to-green-600 text-white py-12 px-6 shadow-lg">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-wide">
            Sağlıklı Alternatifler Rehberi
          </h1>
          <p className="text-lg md:text-xl opacity-90">
            Daha doğal ve sağlıklı bir yaşam için bilinçli tüketim ipuçları
          </p>
        </div>
      </header>

      {/* İçerik */}
      <main className="max-w-5xl mx-auto py-12 px-6">
        {/* Giriş */}
        <section className="mb-12">
          <Card className="border-none shadow-2xl transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
            <CardHeader className="bg-white/80 backdrop-blur-sm p-6 rounded-t-xl">
              <CardTitle className="text-2xl md:text-3xl font-bold text-blue-700">
                Bilinçli Tüketiciler İçin Rehber
              </CardTitle>
              <CardDescription className="mt-2 text-gray-900">
                Günlük hayatımızda kullandığımız ürünlerin içeriklerini bilmek ve doğru tercihler yapmak sağlığımız için önemlidir.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <p className="text-gray-900 mb-4 leading-relaxed">
                Modern yaşamda etrafımız çeşitli kimyasallar ve katkı maddeleriyle dolu ürünlerle çevrili. Bu rehber, zararlı içeriklerden kaçınmak ve daha sağlıklı alternatifleri seçmek için size yol gösterecek kapsamlı bir kaynak sunmaktadır.
              </p>
              <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
                <p className="flex items-start text-blue-800">
                  <AlertTriangle className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-900">
                    Herhangi bir sağlık sorununuz varsa, ürün değişikliklerinden önce sağlık uzmanınıza danışmanızı öneririz.
                  </span>
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Ana Kategoriler */}
        <Tabs defaultValue="food" className="mb-12">
          <TabsList className="grid grid-cols-3 gap-4 mb-8 bg-white/80 backdrop-blur-sm p-4 rounded-lg">
            <TabsTrigger value="food" className="text-black text-base font-semibold px-4 py-2 rounded-lg hover:bg-blue-100">
              Gıda Ürünleri
            </TabsTrigger>
            <TabsTrigger value="cosmetics" className="text-black text-base font-semibold px-4 py-2 rounded-lg hover:bg-blue-100">
              Kozmetik Ürünler
            </TabsTrigger>
            <TabsTrigger value="cleaning" className="text-black text-base font-semibold px-4 py-2 rounded-lg hover:bg-blue-100">
              Temizlik Ürünleri 
            </TabsTrigger>
          </TabsList>
          
          {/* Gıda Ürünleri */}
          <TabsContent value="food">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-green-700">
                    <Leaf className="h-5 w-5 mr-2" /> 
                    Sağlıklı Gıda Alternatifleri
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Rafine Şeker Yerine:</span>
                        <span className="text-gray-500"> Ham bal, akçaağaç şurubu, Hindistan cevizi şekeri, stevia</span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Rafine Un Yerine:</span> 
                        <span className="text-gray-500">Tam buğday unu, badem unu, hindistan cevizi unu, yulaf unu</span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">İşlenmiş Et Yerine:</span>
                        <span className="text-gray-500">Organik tavuk, yumurta, baklagiller, tofu</span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Hazır Atıştırmalıklar Yerine:</span> 
                        <span className="text-gray-500">Kuru meyveler, çiğ kuruyemişler, ev yapımı granola </span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Gazlı İçecekler Yerine:</span> 
                        <span className="text-gray-500">Doğal maden suyu, ev yapımı meyve suları, bitki çayları </span>
                      </div>
                    </li>
                  </ul>
                </CardContent>
              </Card>
              
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-red-700">
                    <AlertTriangle className="h-5 w-5 mr-2" />
                    Kaçınılması Gereken İçerikler
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Yüksek Fruktozlu Mısır Şurubu (HFCS):</span> 
                        <span className="text-gray-500">Obezite ve metabolik sendromla ilişkilendirilmiştir ve kalp hastalıklarına sebep olabilir</span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Trans Yağlar (Kısmen Hidrojenize Yağlar):</span> 
                        <span className="text-gray-500">Kalp hastalığı riskini artırabilir ve alerjilerle ilişkilendirilmiştir</span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Nitratlar ve Nitritler:</span> 
                        <span className="text-gray-500">İşlenmiş etlerde koruyucu olarak kullanılır, kanserle ilişkilendirilmiştir </span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Yapay Renklendiriciler (E102, E110, E133 vb.):</span> 
                        <span className="text-gray-500">Hiperaktivite ve alerjilerle ilişkilendirilmiştir </span>
                      </div>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">BHA ve BHT (E320, E321):</span> 
                        <span className="text-gray-500">Sentetik antioksidanlar, endokrin sistemi bozucu olabilir </span>
                      </div>
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* Kozmetik Ürünler */}
          <TabsContent value="cosmetics">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-green-700">
                    <Leaf className="h-5 w-5 mr-2" />
                    Doğal Kozmetik Alternatifleri
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Sentetik Parfümler Yerine:</span> 
                        <span className="text-gray-500">Esansiyel yağlarla parfümlendirilmiş ürünler  </span>
                      </div>
                    </li>
                    {/* Ek içerikler */}
                  </ul>
                </CardContent>
              </Card>
              
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-red-700">
                    <AlertTriangle className="h-5 w-5 mr-2" />
                    Kozmetikte Kaçınılması Gereken İçerikler
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Parabenler (Methylparaben, Propylparaben):</span> 
                        <span className="text-gray-500">Endokrin sistem bozucu etkisi olabilir  </span>
                      </div>
                    </li>
                    {/* Ek içerikler */}
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* Temizlik Ürünleri */}
          <TabsContent value="cleaning">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-green-700">
                    <Leaf className="h-5 w-5 mr-2" />
                    Doğal Temizlik Alternatifleri
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Kimyasal Çok Amaçlı Temizleyiciler Yerine:</span> 
                        <span className="text-gray-500">Limon Sirke, karbonat ve limon suyu karışımları </span>
                      </div>
                    </li>
                    {/* Ek içerikler */}
                  </ul>
                </CardContent>
              </Card>
              
              <Card className="transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
                <CardHeader className="p-4">
                  <CardTitle className="flex items-center text-xl font-semibold text-red-700">
                    <AlertTriangle className="h-5 w-5 mr-2" />
                    Temizlik Ürünlerinde Kaçınılması Gereken İçerikler
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <ul className="space-y-3">
                    <li className="flex items-start">
                      <AlertTriangle className="h-5 w-5 text-red-500 mr-2 mt-1 flex-shrink-0" />
                      <div>
                        <span className="font-medium text-gray-900">Triklosan:</span> 
                        <span className="text-gray-500">Antibakteriyel özellikli, solunum yolları sorunlarına neden olabilir </span>
                      </div>
                    </li>
                    {/* Ek içerikler */}
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
        
        {/* Bilgilendirici Akordeon */}
        <section className="mb-12">
          <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-blue-700 text-center">
            Etiket Okuma Rehberi
          </h2>
          <LabelGuideAccordion />
        </section>
        
        {/* Alışveriş Rehberi */}
        <section className="mb-12">
          <Card className="border-none shadow-2xl transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
            <CardHeader className="p-6">
              <CardTitle className="flex items-center text-2xl md:text-3xl font-bold text-blue-700">
                <ShoppingBag className="h-6 w-6 mr-2" />
                Bilinçli Alışveriş Rehberi
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-xl md:text-2xl font-semibold mb-3 text-green-700">
                    Alışveriş Yaparken Dikkat Edilecekler
                  </h3>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">Ürün etiketlerini mutlaka okuyun ve içerik listesini kontrol edin</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">İlk 5 içeriğe özellikle dikkat edin, üründe en çok bunlar bulunur</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">Zararlı içerikler hakkında bilgi edinin ve ürünleri karşılaştırın</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">Güvenilir sertifikalara ve ekolojik etiketlere dikkat edin</span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">Sorgulayıcı olun ve pazarlama taktiklerine kanmayın</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-xl md:text-2xl font-semibold mb-3 text-green-700">
                    Güvenilir Sertifikalar
                  </h3>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <Leaf className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">
                        <strong>ICEA, ECOCERT:</strong> Organik kozmetik ürünler için güvenilir sertifikalar
                      </span>
                    </li>
                    <li className="flex items-start">
                      <Leaf className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">
                        <strong>BDIH, COSMOS:</strong> Doğal kozmetik ürünleri için Avrupa standartları
                      </span>
                    </li>
                    <li className="flex items-start">
                      <Leaf className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">
                        <strong>EU Organic Logo:</strong> Avrupa Birliği'nin organik gıda standardı
                      </span>
                    </li>
                    <li className="flex items-start">
                      <Leaf className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">
                        <strong>Cruelty-Free:</strong> Hayvanlar üzerinde test edilmemiş ürünler
                      </span>
                    </li>
                    <li className="flex items-start">
                      <Leaf className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                      <span className="text-gray-900">
                        <strong>Fair Trade:</strong> Adil ticaret koşullarında üretilmiş ürünler
                      </span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
        
        {/* Kaynaklar ve Ek Bilgiler */}
        <section>
          <Card className="bg-blue-50 border-none shadow-2xl transition transform hover:-translate-y-1 hover:shadow-xl rounded-xl">
            <CardHeader className="p-6">
              <CardTitle className="flex items-center text-2xl md:text-3xl font-bold text-blue-700">
                <Heart className="h-6 w-6 mr-2" />
                Sağlıklı Yaşam İpuçları
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <p className="mb-4 text-gray-900 leading-relaxed">
                Ürün seçimlerinizin yanında, aşağıdaki alışkanlıklar da sağlıklı bir yaşam için önemlidir:
              </p>
              <ul className="grid md:grid-cols-2 gap-3">
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">Günde en az 2 litre su için</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">Mevsiminde, yerel ve taze ürünleri tercih edin</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">İşlenmiş gıdalardan uzak durun</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">Düzenli fiziksel aktivite yapın</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">Yeterli ve kaliteli uyku alın</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-1 flex-shrink-0" />
                  <span className="text-gray-900">Stres yönetimi için meditasyon ve nefes egzersizleri yapın</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>
      </main>
        
      {/* Footer */}
      <footer className="bg-gradient-to-r from-blue-700 to-green-600 text-white py-8 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0 text-center md:text-left">
              <h3 className="text-xl md:text-2xl font-bold mb-2">Sağlıklı Ürün Rehberi</h3>
              <p className="opacity-90">Bilinçli tüketim için kapsamlı kaynak</p>
            </div>
            <div className="text-center">
              <p>&copy; {new Date().getFullYear()} Sağlıklı Ürün Rehberi. Tüm hakları saklıdır.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
