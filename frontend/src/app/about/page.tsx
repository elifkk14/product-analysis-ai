import React from 'react';
import Link from 'next/link';

export default function About() {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
          Hakkımızda
        </h1>
        <p className="mt-4 text-xl text-gray-500">
          Kullandığınız ürünler hakkında bilinçli kararlar vermenize yardımcı oluyoruz
        </p>
      </div>

      <div className="prose prose-lg mx-auto">
        <p>
          Platformumuz, tüketicilerin ürün içeriklerini potansiyel sağlık riskleri açısından analiz etmelerine olanak tanır.
          Gelişmiş algoritmalar ve kapsamlı veritabanları kullanarak, günlük ürünlerdeki zararlı maddeleri ve alerjenleri belirliyoruz.
        </p>

        <h2 className="text-2xl font-bold mt-8 mb-4">Misyonumuz</h2>
        <p>
          Herkesin günlük olarak kullandığı ürünlerin içeriğini bilme hakkına sahip olduğuna inanıyoruz. 
          Misyonumuz, daha sağlıklı seçimler yapmanıza yardımcı olmak için ürün içerikleri hakkında şeffaf, 
          erişilebilir bilgiler sunmaktır.
        </p>

        <h2 className="text-2xl font-bold mt-8 mb-4">Nasıl Çalışır</h2>
        <ul className="list-disc pl-6 space-y-2 my-4">
          <li>Ürün içeriklerini yükleyin veya girin</li>
          <li>Sistemimiz her bileşeni analiz eder</li>
          <li>Potansiyel endişeleri vurgulayan detaylı bir rapor alın</li>
          <li>Sağlık profilinize göre kişiselleştirilmiş öneriler alın</li>
        </ul>

        <div className="bg-blue-50 p-6 rounded-lg mt-8">
          <h2 className="text-2xl font-bold mb-4 text-blue-700">Ücretsiz Analizinizi Başlatın</h2>
          <p className="mb-4">
            Ürünlerinizde gerçekte ne olduğunu keşfetmeye hazır mısınız? Analiz aracımızı hemen deneyin.
          </p>
          <Link href="/analysis">
            <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Ürün Analizi
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}