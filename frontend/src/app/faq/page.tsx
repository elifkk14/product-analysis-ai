"use client";
import Link from 'next/link';
import React, { useState } from 'react';

interface FAQItemProps {
  question: string;
  answer: string;
}

const FAQItem: React.FC<FAQItemProps> = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="group relative mb-4 cursor-pointer rounded-xl transition-all duration-200 hover:bg-gray-50/50">
      <button
        className="flex w-full items-center justify-between p-6"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-left text-lg font-semibold text-gray-900">{question}</span>
        <div className="ml-4 flex h-8 w-8 items-center justify-center rounded-full bg-blue-100/50 transition-all group-hover:bg-blue-200/50">
          <svg
            className={`h-5 w-5 text-blue-600 transition-transform duration-300 ${
              isOpen ? 'rotate-45' : 'rotate-0'
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
        </div>
      </button>
      <div
        className={`overflow-hidden px-6 transition-all duration-300 ease-in-out ${
          isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <p className="pb-6 text-gray-600">{answer}</p>
      </div>
    </div>
  );
};

export default function FAQ() {
  // SSS verileri
  const faqData = [
    {
      question: "Ürün analizi nasıl çalışır?",
      answer: "Ürün içeriklerini sistemimize girdiğinizde, veritabanımızdaki bilgilerle karşılaştırarak her bir içeriğin potansiyel sağlık risklerini değerlendiriyoruz. Analiz sonunda size ayrıntılı bir rapor sunuyoruz."
    },
    {
      question: "Hangi ürünleri analiz edebilirim?",
      answer: "Kozmetik ürünleri, temizlik ürünleri, kişisel bakım ürünleri ve besin takviyeleri gibi birçok ürün kategorisini analiz edebilirsiniz. Ürün etiketindeki içerik listesini girmeniz yeterlidir."
    },
    {
      question: "Analiz sonuçları ne kadar güvenilir?",
      answer: "Sonuçlarımız güncel bilimsel araştırmalara ve uluslararası güvenlik standartlarına dayanmaktadır. Bununla birlikte, analiz sonuçları tıbbi tavsiye yerine geçmez ve her zaman profesyonel sağlık danışmanlarına başvurmanızı öneririz."
    },
    {
      question: "Ürün analizi ücretli mi?",
      answer: "Temel analiz hizmetimiz ücretsizdir. Daha detaylı raporlar, kişiselleştirilmiş tavsiyeler ve premium özellikler için aylık abonelik planlarımız bulunmaktadır."
    },
    {
      question: "Kişisel bilgilerim nasıl korunuyor?",
      answer: "Gizliliğinizi çok önemsiyoruz. Kişisel verileriniz en yüksek güvenlik standartlarına göre şifrelenir ve saklanır. Verilerinizi üçüncü taraflarla asla paylaşmıyoruz ve yalnızca size daha iyi hizmet sunmak için kullanıyoruz."
    },
    {
      question: "Analiz sonucunda zararlı bir içerik tespit edilirse ne yapmalıyım?",
      answer: "Analiz sonuçlarında potansiyel zararlı içerikler tespit edilirse, alternatif ürün önerileri sunuyoruz. Cilt reaksiyonu veya alerjik tepki yaşıyorsanız, kullanımı hemen bırakmanızı ve bir sağlık uzmanına danışmanızı öneririz."
    }
  ];

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 p-8 shadow-xl sm:p-12">
        <div className="mb-12 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Sık Sorulan Sorular
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-xl text-gray-600">
            Ürün analizi süreci hakkında merak ettikleriniz
          </p>
        </div>

        <div className="space-y-4 divide-y divide-gray-200/60">
          {faqData.map((faq, index) => (
            <FAQItem key={index} question={faq.question} answer={faq.answer} />
          ))}
        </div>

        <div className="mt-12 rounded-xl bg-white p-8 shadow-lg">
          <div className="flex flex-col items-center gap-6 text-center md:flex-row md:text-left">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-gray-900">Hala sorunuz mu var?</h2>
              <p className="mt-2 text-gray-600">
                Uzman ekibimiz size en kısa sürede dönüş yapacaktır
              </p>
            </div>
            <Link href="/contact">
            <button className="transform rounded-lg bg-blue-600 px-8 py-3 font-medium text-white transition-all hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-200 focus:outline-none active:scale-95">
              İletişime Geçin
              <span aria-hidden="true" className="ml-2">
                →
              </span>
            </button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}