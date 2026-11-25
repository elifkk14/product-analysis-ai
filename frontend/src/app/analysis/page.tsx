"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

type AnalysisResult = {
  harmful_ingredients: string[];
  safe_ingredients: string[];
  toxicity_level: "Low" | "Medium" | "High";
  models_used?: string[];
};

export default function Analysis() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(false);

  // Dark mode toggle - localStorage kullanımı kaldırıldı
  useEffect(() => {
    setDarkMode(false); // Varsayılan olarak light mode
  }, []);

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    
    if (newDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const handleSubmit = () => {
    analyzeContent();
  };

  const exampleAnalyses = [
    "Su, Sodyum Lauril Sülfat, Propilen Glikol, Parfüm",
    "Aloe Vera, Gliserin, Argan Yağı, Zeytinyağı, Shea Yağı",
    "Titanyum Dioksit, Paraben, Formaldehit, Mineral Yağ"
  ];

  const analyzeContent = async () => {
    if (!inputText.trim()) {
      setError("Lütfen ürün içeriklerini girin!");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    // Simulated API call with realistic delay
    setTimeout(() => {
      // Mock analysis result
      const mockResult: AnalysisResult = {
        harmful_ingredients: inputText.toLowerCase().includes("paraben") ? ["Paraben", "Formaldehit"] : 
                           inputText.toLowerCase().includes("sülfat") ? ["Sodyum Lauril Sülfat"] : [],
        safe_ingredients: ["Su", "Gliserin", "Aloe Vera", "Argan Yağı"].filter(ing => 
          inputText.toLowerCase().includes(ing.toLowerCase())),
        toxicity_level: inputText.toLowerCase().includes("paraben") ? "High" : 
                       inputText.toLowerCase().includes("sülfat") ? "Medium" : "Low",
        models_used: ["GPT-4", "BERT"]
      };
      
      setResult(mockResult);
      setHistory(prev => [...prev.filter(item => item !== inputText).slice(0, 3), inputText]);
      setLoading(false);
    }, 2000);
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const renderAnalysisResults = () => {
    if (!result) return null;

    const toxicityConfig = {
      High: { 
        color: 'from-red-500 to-pink-600', 
        bg: 'bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20',
        icon: '🚨',
        text: 'Yüksek Risk'
      },
      Medium: { 
        color: 'from-amber-500 to-orange-600', 
        bg: 'bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20',
        icon: '⚡',
        text: 'Orta Risk'
      },
      Low: { 
        color: 'from-emerald-500 to-green-600', 
        bg: 'bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20',
        icon: '✨',
        text: 'Düşük Risk'
      }
    };

    const config = toxicityConfig[result.toxicity_level];

    return (
      <motion.div 
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="mt-8 relative"
      >
        {/* Glow effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-3xl blur-xl"></div>
        
        <div className="relative backdrop-blur-xl bg-white/70 dark:bg-gray-800/70 p-8 rounded-3xl border border-white/20 shadow-2xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center justify-between mb-8"
          >
            <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              📊 Analiz Sonuçları
            </h2>
            <motion.button
              whileHover={{ scale: 1.1, rotate: 180 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setResult(null)}
              className="p-3 rounded-full bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600 hover:from-red-100 hover:to-red-200 transition-all duration-300"
            >
              ✕
            </motion.button>
          </motion.div>

          {/* Toxicity Level - Prominent Display */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, type: "spring" }}
            className={`${config.bg} rounded-2xl p-6 mb-8 border border-white/30`}
          >
            <div className="text-center">
              <div className={`inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r ${config.color} text-white font-bold text-lg shadow-lg`}>
                <span className="text-2xl mr-3">{config.icon}</span>
                {config.text}
              </div>
              <p className="mt-3 text-gray-600 dark:text-gray-300 font-medium">
                Genel Toksisite Değerlendirmesi
              </p>
            </div>
          </motion.div>

          {/* Results Grid */}
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Zararlı İçerikler */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-red-400/20 to-pink-400/20 rounded-2xl blur-lg group-hover:blur-xl transition-all duration-300"></div>
              <div className="relative backdrop-blur-sm bg-gradient-to-br from-red-50/80 to-pink-50/80 dark:from-red-900/20 dark:to-pink-900/20 rounded-2xl p-6 border border-red-200/30 dark:border-red-800/30">
                <div className="flex items-center mb-4">
                  <div className="p-3 rounded-full bg-gradient-to-r from-red-500 to-pink-600 text-white mr-4">
                    ⚠️
                  </div>
                  <h3 className="text-xl font-bold text-red-700 dark:text-red-400">
                    Zararlı İçerikler
                  </h3>
                </div>
                
                {result.harmful_ingredients.length > 0 ? (
                  <div className="space-y-3">
                    {result.harmful_ingredients.map((ingredient, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 + index * 0.1 }}
                        className="flex items-center p-3 bg-white/50 dark:bg-gray-800/50 rounded-xl border border-red-200/50"
                      >
                        <span className="text-red-500 mr-3 text-lg">❌</span>
                        <span className="text-red-700 dark:text-red-300 font-medium">{ingredient}</span>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="text-6xl mb-4">🎉</div>
                    <p className="text-green-600 dark:text-green-400 font-bold text-lg">
                      Zararlı içerik bulunamadı!
                    </p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Güvenli İçerikler */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 to-green-400/20 rounded-2xl blur-lg group-hover:blur-xl transition-all duration-300"></div>
              <div className="relative backdrop-blur-sm bg-gradient-to-br from-emerald-50/80 to-green-50/80 dark:from-emerald-900/20 dark:to-green-900/20 rounded-2xl p-6 border border-emerald-200/30 dark:border-emerald-800/30">
                <div className="flex items-center mb-4">
                  <div className="p-3 rounded-full bg-gradient-to-r from-emerald-500 to-green-600 text-white mr-4">
                    ✅
                  </div>
                  <h3 className="text-xl font-bold text-emerald-700 dark:text-emerald-400">
                    Güvenli İçerikler
                  </h3>
                </div>
                
                {result.safe_ingredients.length > 0 ? (
                  <div className="space-y-3">
                    {result.safe_ingredients.map((ingredient, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 + index * 0.1 }}
                        className="flex items-center p-3 bg-white/50 dark:bg-gray-800/50 rounded-xl border border-emerald-200/50"
                      >
                        <span className="text-emerald-500 mr-3 text-lg">✅</span>
                        <span className="text-emerald-700 dark:text-emerald-300 font-medium">{ingredient}</span>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="text-6xl mb-4">😔</div>
                    <p className="text-gray-600 dark:text-gray-400 font-medium">
                      Güvenli içerik bulunamadı
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {/* Models Used */}
          {result.models_used && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mt-8 text-center"
            >
              <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-full border border-blue-200/50">
                <span className="text-blue-600 dark:text-blue-400 mr-2">🧠</span>
                <span className="text-sm text-gray-600 dark:text-gray-300">
                  Kullanılan AI Modelleri: {result.models_used.join(" + ")}
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900 transition-all duration-500">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,119,198,0.1),transparent_50%)]"></div>
      
      <div className="relative max-w-6xl mx-auto p-6 md:p-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-between items-center mb-8"
        >
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
              🧪 Akıllı İçerik Analizörü
            </h1>
            <p className="text-gray-600 dark:text-gray-400 text-lg">
              Yapay zeka ile ürün içeriklerini analiz edin
            </p>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={toggleDarkMode}
            className="p-4 rounded-2xl bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600 shadow-lg hover:shadow-xl transition-all"
          >
            <span className="text-2xl">{darkMode ? "☀️" : "🌙"}</span>
          </motion.button>
        </motion.div>
        
        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid md:grid-cols-5 gap-4 mb-8"
        >
          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => alert("Barkod tarayıcı açılacak")}
            className="md:col-span-4 relative group overflow-hidden bg-gradient-to-r from-indigo-500 to-purple-600 text-white py-4 px-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center justify-center font-bold text-lg">
              <span className="mr-3 text-2xl">📷</span> Barkod Tarayıcı
            </div>
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05, rotate: 5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => alert("Yardım sayfası açılacak")}
            className="relative group bg-gradient-to-r from-gray-400 to-gray-500 text-white py-4 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl"></div>
            <div className="relative text-2xl">❓</div>
          </motion.button>
        </motion.div>

        {/* Main Form */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="relative"
        >
          {/* Glass effect background */}
          <div className="absolute inset-0 bg-gradient-to-r from-white/40 to-white/20 dark:from-gray-800/40 dark:to-gray-700/20 backdrop-blur-xl rounded-3xl border border-white/20"></div>
          
          <div className="relative p-8 space-y-6">
            <div>
              <label htmlFor="ingredients" className="block text-gray-700 dark:text-gray-300 mb-3 font-bold text-lg">
                İçerik Listesi:
              </label>
              <div className="relative">
                <textarea
                  id="ingredients"
                  className="w-full p-6 rounded-2xl border border-gray-200/50 dark:border-gray-600/50 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm focus:outline-none focus:ring-4 focus:ring-blue-500/30 focus:border-blue-500/50 transition-all duration-300 text-lg resize-none"
                  rows={6}
                  placeholder="İçerikleri virgülle ayırarak girin...&#10;Örnek: Sodyum Lauril Sülfat, Paraben, Gliserin"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={loading}
                />
                <div className="absolute bottom-4 right-4 text-gray-400 text-sm">
                  {inputText.length}/500
                </div>
              </div>
            </div>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  className="p-4 rounded-2xl bg-gradient-to-r from-red-100 to-pink-100 dark:from-red-900/30 dark:to-pink-900/30 border border-red-200/50 flex items-center"
                >
                  <span className="text-2xl mr-3">⚠️</span>
                  <span className="text-red-700 dark:text-red-300 font-medium">{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button
              onClick={handleSubmit}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              disabled={loading}
              className="w-full relative group overflow-hidden bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white py-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 text-xl font-bold"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative flex items-center justify-center">
                {loading ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-6 h-6 border-2 border-white border-t-transparent rounded-full mr-3"
                    />
                    İçerikler Analiz Ediliyor...
                  </>
                ) : (
                  <>
                    <span className="mr-3 text-2xl">🔍</span>
                    Yapay Zeka ile Analiz Et
                  </>
                )}
              </div>
            </motion.button>
          </div>
        </motion.div>

        {/* Example Ingredients */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-8"
        >
          <h3 className="text-gray-600 dark:text-gray-400 font-bold mb-4 text-lg">Örnek İçerikler:</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {exampleAnalyses.map((example, index) => (
              <motion.button
                key={index}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setInputText(example)}
                className="p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-2xl border border-gray-200/50 hover:border-blue-300/50 transition-all duration-300 text-left shadow-md hover:shadow-lg"
              >
                <div className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                  {example}
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>

        <AnimatePresence>
          {renderAnalysisResults()}
        </AnimatePresence>

        {/* History Section */}
        <AnimatePresence>
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-12"
            >
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-2xl font-bold text-gray-700 dark:text-gray-300">
                  📚 Son Analizler
                </h3>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={clearHistory}
                  className="px-4 py-2 bg-gradient-to-r from-red-400 to-pink-500 text-white rounded-xl text-sm font-medium hover:shadow-lg transition-all"
                >
                  Temizle
                </motion.button>
              </div>
              
              <div className="grid gap-4">
                {history.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ x: 10, scale: 1.01 }}
                    className="p-4 bg-gradient-to-r from-white/60 to-gray-50/60 dark:from-gray-800/60 dark:to-gray-700/60 backdrop-blur-sm rounded-2xl border border-gray-200/30 cursor-pointer transition-all duration-300 flex justify-between items-center group"
                    onClick={() => setInputText(item)}
                  >
                    <span className="text-gray-700 dark:text-gray-300 truncate flex-1 mr-4">
                      {item}
                    </span>
                    <motion.span
                      whileHover={{ x: 5 }}
                      className="text-blue-500 text-xl group-hover:text-purple-500 transition-colors"
                    >
                      →
                    </motion.span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}