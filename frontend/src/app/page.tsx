
"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ModernHome() {
  const [loading, setLoading] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [scanMode, setScanMode] = useState(false);
  const [recentSearches, setRecentSearches] = useState([
    "Protein Bar", "Şampuan", "Cilt Kremi", "Çikolata"
  ]); 
  const [popularProducts, setPopularProducts] = useState([
    { name: "Dove Şampuan", safeScore: 85, image: "https://cdn.dsmcdn.com/ty49/product/media/images/20210106/7/47628565/66715390/1/1_org_zoom.jpg" },
    { name: "Nivea El Kremi", safeScore: 92, image: "https://cdn.dsmcdn.com/mnresize/400/-/ty467/product/media/images/20220701/14/133393185/437619593/1/1_org.jpg" },
    { name: "Ülker Çikolata", safeScore: 75, image: "https://reimg-carrefour.mncdn.com/mnresize/600/600/productimage/30098860/30098860_0_MC/8797015900210_1549347753689.jpg" },
    { name: "Duru Sabun", safeScore: 90, image: "https://images.migrosone.com/sanalmarket/product/29200037/29200037-4fe488-1650x1650.jpg" }
  ]);

  // Updated color palette to match the cosmetics/sustainable living aesthetic
  const colors = {
    primary: '#B8C4B9',       // Soft sage green
    primaryDark: '#8A9B8D',   // Darker muted green
    secondary: '#F5F5F5',     // Light cream background
    accent: '#D4B483',        // Warm beige accent
    textDark: '#6B5B4D',      // Muted brown text
    textLight: '#FFFFFF',     // White text
    highlight: '#E8D9C5',     // Light highlight
    sale: '#D4A59A',          // Soft coral for sale/alert items
    background: '#E8E8E8',    // Light gray background
  };

  const handleStartAnalysis = () => {
    if (searchInput.trim()) {
      setLoading(true);
      setTimeout(() => {
        alert(`Analiz başlatılıyor: ${searchInput}`);
        setLoading(false);
      }, 2000);
    } else {
      setIsModalOpen(true);
    }
  };

  const handleScanMode = () => {
    setScanMode(true);
    setTimeout(() => {
      setScanMode(false);
      setSearchInput("Dove Şampuan (Scanned)");
    }, 2000);
  };

  const handleQuickSearch = (product: string) => {
    setSearchInput(product);
    setLoading(true);
    setTimeout(() => {
      alert(`Analiz başlatılıyor: ${product}`);
      setLoading(false);
    }, 2000);
  };

  const stats = [
    { label: "Ürün Analizi", value: "10B+", icon: "🔬" },
    { label: "Zararlı İçerikler", value: "500+", icon: "⚠️" },
    { label: "Memnun Kullanıcı", value: "1000+", icon: "😊" },
  ];

  const testimonials = [
    { name: "Elif T.", review: "Bu uygulama ile zararlı katkı maddelerini kolayca tespit edebiliyorum!", avatar: "👩‍💼" },
    { name: "Ahmet Y.", review: "Market alışverişlerinde çok işime yarıyor!", avatar: "👨‍💻" },
    { name: "Zeynep K.", review: "Çocuğumun alerjisi var ve bu uygulama sayesinde güvenli ürünleri bulabiliyorum.", avatar: "👩‍⚕️" },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#F5F5F5] via-[#E8D9C5] to-[#F5F5F5]">
      
      {/* Modern Hero Section */}
      <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Simplified Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-[#8A9B8D] to-[#B8C4B9]"></div>
          <motion.div
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.1, 0.2, 0.1]
            }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute -top-1/2 -right-1/2 w-full h-full bg-[#D4B483]/20 rounded-full blur-3xl"
          />
        </div>

        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center text-white">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <h1 className="text-6xl md:text-7xl font-black mb-6 bg-gradient-to-r from-white to-[#E8D9C5] bg-clip-text text-transparent">
              Ürün Güvenliği
            </h1>
            <h2 className="text-4xl md:text-5xl font-bold mb-8 text-[#F5F5F5]">
              Yeniden Tanımlandı
            </h2>
            <p className="text-xl md:text-2xl text-[#E8D9C5] max-w-3xl mx-auto leading-relaxed">
              AI destekli analiz ile ürünlerinizin içeriğini keşfedin ve sağlığınız için en güvenli seçimleri yapın
            </p>
          </motion.div>

          {/* Modern Search Interface */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="max-w-4xl mx-auto mb-12"
          >
            <div className="backdrop-blur-xl bg-white/20 rounded-3xl p-8 border border-white/30 shadow-sm">
              <div className="flex flex-col md:flex-row gap-4 mb-6">
                <div className="flex-1 relative">
                  <input 
                    type="text" 
                    value={searchInput}
                    onChange={(e) => setSearchInput(e.target.value)}
                    placeholder="Ürün adı, barkod veya içerik listesi..." 
                    className="w-full p-6 bg-white/90 backdrop-blur-md rounded-2xl text-[#6B5B4D] placeholder-[#B8C4B9] border-0 focus:ring-4 focus:ring-[#8A9B8D]/50 outline-none text-lg font-medium shadow-sm"
                  />
                  {searchInput && (
                    <button 
                      onClick={() => setSearchInput("")}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-[#B8C4B9] hover:text-[#6B5B4D] text-2xl font-bold"
                    >
                      ×
                    </button>
                  )}
                </div>
                
                <div className="flex gap-3">
                  <motion.button 
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleScanMode}
                    className="bg-gradient-to-r from-[#8A9B8D] to-[#B8C4B9] text-white px-8 py-6 rounded-2xl font-bold text-lg shadow-sm hover:shadow-md transition-all duration-300 flex items-center gap-3"
                  >
                    {scanMode ? (
                      <motion.span 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="text-2xl"
                      >
                        📷
                      </motion.span>
                    ) : (
                      <span className="text-2xl">📷</span>
                    )}
                    {scanMode ? "Taranıyor..." : "Tara"}
                  </motion.button>
                  
                  <motion.button 
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleStartAnalysis}
                    className="bg-gradient-to-r from-[#6B5B4D] to-[#D4B483] text-white px-8 py-6 rounded-2xl font-bold text-lg shadow-sm hover:shadow-md transition-all duration-300 flex items-center gap-3"
                  >
                    {loading ? (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-6 h-6 border-3 border-white border-t-transparent rounded-full"
                      />
                    ) : (
                      <span className="text-2xl">🔍</span>
                    )}
                    Analiz Et
                  </motion.button>
                </div>
              </div>

              {/* Quick Search Tags */}
              <div className="flex flex-wrap justify-center gap-3">
                {recentSearches.map((search, index) => (
                  <motion.button
                    key={index}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleQuickSearch(search)}
                    className="bg-white/20 backdrop-blur-md border border-white/30 px-6 py-3 rounded-full text-sm font-medium text-white hover:bg-white/30 transition-all duration-300"
                  >
                    {search}
                  </motion.button>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-[#E8D9C5]"
          >
            <p className="mb-4">Örnek kullanımlar:</p>
            <div className="text-sm space-y-1">
              <p>• "Dove Şampuan" veya "L'Oreal Paris Foundation"</p>
              <p>• "Sodyum Lauril Sülfat, Parabens, Formaldehit"</p>
              <p>• Barkod numarası: "8690536127890"</p>
            </div>
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2 text-white/70 cursor-pointer"
          onClick={() => window.scrollTo({ top: window.innerHeight, behavior: "smooth" })}
        >
          <div className="text-4xl">↓</div>
          <p className="text-sm mt-2">Keşfetmeye devam edin</p>
        </motion.div>
      </div>

      {/* Stats Section */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-[#F5F5F5]"
      >
        <div className="max-w-6xl mx-auto px-6">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold text-center mb-16 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] bg-clip-text text-transparent"
          >
            Güvenilir Veriler
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -10, scale: 1.05 }}
                className="text-center p-8 rounded-3xl bg-white shadow-sm border border-[#E8D9C5]"
              >
                <div className="text-6xl mb-4 text-[#6B5B4D]">{stat.icon}</div>
                <div className="text-5xl font-black text-[#6B5B4D] mb-4">{stat.value}</div>
                <div className="text-xl text-[#8A9B8D] font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Popular Products */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-[#E8D9C5]"
      >
        <div className="max-w-6xl mx-auto px-6">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold text-center mb-4 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] bg-clip-text text-transparent"
          >
            Popüler Ürünler
          </motion.h2>
          <motion.p 
            variants={itemVariants}
            className="text-xl text-[#6B5B4D] text-center mb-16"
          >
            En çok analiz edilen ürünleri keşfedin
          </motion.p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {popularProducts.map((product, index) => (
              <motion.div 
                key={index}
                variants={itemVariants}
                whileHover={{ y: -10, scale: 1.02 }}
                className="group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-300 cursor-pointer border border-[#E8D9C5]"
                onClick={() => handleQuickSearch(product.name)}
              >
                <div className="h-48 bg-gradient-to-br from-[#F5F5F5] to-[#E8D9C5] flex items-center justify-center p-6">
                  <img 
                    src={product.image} 
                    alt={product.name} 
                    className="h-32 w-32 object-contain group-hover:scale-110 transition-transform duration-300" 
                  />
                </div>
                <div className="p-6">
                  <h3 className="font-bold text-lg text-[#6B5B4D] mb-3 group-hover:text-[#8A9B8D] transition-colors">
                    {product.name}
                  </h3>
                  <div className="flex items-center justify-between">
                    <span className="text-[#8A9B8D] font-medium">Güvenlik Skoru</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${
                        product.safeScore >= 85 ? 'bg-[#8A9B8D]' : 
                        product.safeScore >= 70 ? 'bg-[#D4B483]' : 'bg-[#6B5B4D]'
                      }`}></div>
                      <span className={`font-bold text-lg ${
                        product.safeScore >= 85 ? 'text-[#8A9B8D]' : 
                        product.safeScore >= 70 ? 'text-[#D4B483]' : 'text-[#6B5B4D]'
                      }`}>
                        {product.safeScore}%
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* How It Works */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-white"
      >
        <div className="max-w-6xl mx-auto px-6">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold text-center mb-16 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] bg-clip-text text-transparent"
          >
            Nasıl Çalışır?
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { 
                step: "01", 
                title: "Ürünü Girin", 
                desc: "Ürün adını yazın, barkod tarayın veya içerik listesini paylaşın", 
                icon: "📝",
                color: "from-[#8A9B8D] to-[#B8C4B9]"
              },
              { 
                step: "02", 
                title: "AI Analizi", 
                desc: "Gelişmiş yapay zeka algoritmaları içerikleri detaylı analiz eder", 
                icon: "🤖",
                color: "from-[#D4B483] to-[#E8D9C5]"
              },
              { 
                step: "03", 
                title: "Sonuçları Görün", 
                desc: "Kapsamlı rapor ve önerilerle güvenli tercihler yapın", 
                icon: "📊",
                color: "from-[#6B5B4D] to-[#8A9B8D]"
              }
            ].map((item, index) => (
              <motion.div 
                key={index}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="relative"
              >
                <div className="text-center p-8 rounded-3xl bg-white shadow-sm border border-[#E8D9C5] h-full">
                  <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-r ${item.color} flex items-center justify-center text-3xl shadow-sm`}>
                    {item.icon}
                  </div>
                  <div className={`text-6xl font-black mb-4 bg-gradient-to-r ${item.color} bg-clip-text text-transparent`}>
                    {item.step}
                  </div>
                  <h3 className="text-2xl font-bold text-[#6B5B4D] mb-4">{item.title}</h3>
                  <p className="text-[#8A9B8D] leading-relaxed">{item.desc}</p>
                </div>
                
                {index < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 text-4xl text-[#D4B483]">
                    →
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Features Grid */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-[#F5F5F5]"
      >
        <div className="max-w-6xl mx-auto px-6">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold text-center mb-16 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] bg-clip-text text-transparent"
          >
            Güçlü Özellikler
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { title: "AI Destekli Analiz", desc: "Makine öğrenmesi ile hassas değerlendirme", icon: "🧠", color: "from-[#8A9B8D] to-[#B8C4B9]" },
              { title: "Anlık Barkod Tarama", desc: "Kameranızla saniyeler içinde sonuç", icon: "📱", color: "from-[#D4B483] to-[#E8D9C5]" },
              { title: "Kapsamlı Veritabanı", desc: "500+ zararlı madde ve binlerce ürün", icon: "🗃️", color: "from-[#6B5B4D] to-[#8A9B8D]" },
              { title: "Kişisel Öneriler", desc: "Size özel alternativ ürün önerileri", icon: "💡", color: "from-[#B8C4B9] to-[#8A9B8D]" },
              { title: "Topluluk Yorumları", desc: "Gerçek kullanıcı deneyimleri", icon: "💬", color: "from-[#E8D9C5] to-[#D4B483]" },
              { title: "Bilimsel Kaynaklar", desc: "Uzman araştırmalarına dayalı veriler", icon: "🔬", color: "from-[#8A9B8D] to-[#6B5B4D]" },
              { title: "Offline Çalışma", desc: "İnternet olmadan da kullanılabilir", icon: "📡", color: "from-[#B8C4B9] to-[#8A9B8D]" },
              { title: "Çoklu Dil Desteği", desc: "Türkçe, İngilizce ve daha fazlası", icon: "🌍", color: "from-[#D4B483] to-[#6B5B4D]" }
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ scale: 1.05, y: -5 }}
                className="p-6 bg-white rounded-3xl shadow-sm border border-[#E8D9C5] text-center hover:shadow-md transition-all duration-300"
              >
                <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-r ${feature.color} flex items-center justify-center text-2xl shadow-sm`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-[#6B5B4D] mb-3">{feature.title}</h3>
                <p className="text-[#8A9B8D] text-sm leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Testimonials */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-[#E8D9C5]"
      >
        <div className="max-w-6xl mx-auto px-6">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold text-center mb-16 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] bg-clip-text text-transparent"
          >
            Kullanıcı Deneyimleri
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="p-8 bg-white rounded-3xl shadow-sm border border-[#E8D9C5]"
              >
                <div className="text-6xl mb-6 text-center text-[#6B5B4D]">{testimonial.avatar}</div>
                <p className="text-[#6B5B4D] italic text-lg leading-relaxed mb-6">
                  "{testimonial.review}"
                </p>
                <div className="flex items-center justify-center">
                  <div className="text-center">
                    <div className="font-bold text-[#6B5B4D] text-lg">{testimonial.name}</div>
                    <div className="flex text-[#D4B483] text-xl justify-center mt-2">
                      ★★★★★
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* CTA Section */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-20 bg-gradient-to-br from-[#6B5B4D] to-[#8A9B8D] text-white relative overflow-hidden"
      >
        <div className="absolute inset-0">
          <motion.div
            animate={{ 
              scale: [1, 1.3, 1],
              rotate: [0, 360],
              opacity: [0.1, 0.3, 0.1]
            }}
            transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
            className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-br from-[#B8C4B9]/20 to-[#8A9B8D]/20 rounded-full blur-3xl"
          />
        </div>
        
        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <motion.h2 
            variants={itemVariants}
            className="text-5xl font-bold mb-6"
          >
            Sağlıklı Yaşam Başlıyor
          </motion.h2>
          <motion.p 
            variants={itemVariants}
            className="text-xl text-[#E8D9C5] mb-12 leading-relaxed"
          >
            Günlük kullandığınız ürünlerin güvenliğinden emin olun. Yeni zararlı içerikler ve sağlıklı alternatifler hakkında güncel kalın.
          </motion.p>
          
          <motion.div 
            variants={itemVariants}
            className="max-w-lg mx-auto"
          >
            <div className="flex flex-col md:flex-row gap-4 p-2 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20">
              <input 
                type="email" 
                placeholder="E-posta adresinizi girin" 
                className="flex-1 p-4 bg-white/90 backdrop-blur-md rounded-xl text-[#6B5B4D] placeholder-[#8A9B8D] border-0 focus:ring-4 focus:ring-[#8A9B8D]/50 outline-none font-medium"
              />
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] text-white px-8 py-4 rounded-xl font-bold shadow-sm hover:shadow-md transition-all duration-300"
              >
                Hemen Başla
              </motion.button>
            </div>
            <p className="text-[#E8D9C5] text-sm mt-4">
              Gizliliğinizi koruyoruz. İstediğiniz zaman abonelikten çıkabilirsiniz.
            </p>
          </motion.div>
        </div>
      </motion.div>

      {/* Enhanced Modal */}
      <AnimatePresence>
        {isModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setIsModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-3xl p-8 shadow-sm max-w-md w-full border border-[#E8D9C5]"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="text-center">
                <div className="w-20 h-20 bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] rounded-full flex items-center justify-center mx-auto mb-6 text-3xl text-white">
                  ⚠️
                </div>
                <h3 className="text-2xl font-bold text-[#6B5B4D] mb-4">Analiz Başlatılamıyor</h3>
                <p className="text-[#8A9B8D] mb-8 leading-relaxed">
                  Lütfen analiz etmek istediğiniz ürün adını, barkod numarasını veya içerik listesini girin.
                </p>
                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsModalOpen(false)} 
                  className="bg-gradient-to-r from-[#6B5B4D] to-[#8A9B8D] text-white px-8 py-4 rounded-xl font-bold shadow-sm hover:shadow-md transition-all duration-300"
                >
                  Anladım
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}