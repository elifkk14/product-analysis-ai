"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FiSend, FiCheckCircle, FiAlertCircle } from "react-icons/fi";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [status, setStatus] = useState<"idle" | "sending" | "success" | "error">("idle");

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("sending");

    try {
      // API entegrasyon örneği:
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error("Gönderim hatası");
      
      setStatus("success");
      setFormData({ name: "", email: "", subject: "", message: "" });
      setTimeout(() => setStatus("idle"), 3000);
    } catch (error) {
      setStatus("error");
      setTimeout(() => setStatus("idle"), 3000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-indigo-50">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-16 px-6 shadow-xl"
      >
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">
            Bizimle İletişime Geçin
          </h1>
          <p className="text-lg md:text-xl text-blue-100 max-w-2xl mx-auto">
            Sorularınızı yanıtlamaktan, önerilerinizi duymaktan ve iş birliği fırsatlarını
            değerlendirmekten mutluluk duyarız.
          </p>
        </div>
      </motion.header>

      {/* Ana İçerik */}
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="max-w-4xl mx-auto p-6 -mt-12"
      >
        <div className="bg-white rounded-2xl shadow-2xl p-8 md:p-12 space-y-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Mesajınızı Bırakın
            </h2>
            <p className="text-gray-600 max-w-lg mx-auto">
              24 saat içinde ekip üyelerimizden biri sizinle iletişime geçecektir
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Adınız *
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 rounded-lg bg-white border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="John Doe"
                />
              </div>
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Email Adresiniz *
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 rounded-lg bg-white border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="johndoe@example.com"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Konu *
              </label>
              <input
                type="text"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
                className="w-full px-4 py-3 rounded-lg bg-white border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="Nasıl yardımcı olabiliriz?"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                Mesajınız *
              </label>
              <textarea
                name="message"
                rows={5}
                value={formData.message}
                onChange={handleChange}
                required
                className="w-full px-4 py-3 rounded-lg bg-white border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="Detaylı bilgi verirseniz size daha iyi yardımcı olabiliriz..."
              />
            </div>

            <div className="pt-6">
              <button
                type="submit"
                disabled={status === "sending"}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-6 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-70"
              >
                {status === "sending" ? (
                  <>
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Gönderiliyor...
                  </>
                ) : (
                  <>
                    <FiSend className="w-5 h-5" />
                    Mesajı Gönder
                  </>
                )}
              </button>

              {status === "success" && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-green-50 text-green-700 rounded-lg flex items-center gap-3"
                >
                  <FiCheckCircle className="w-5 h-5 flex-shrink-0" />
                  <span>Mesajınız başarıyla iletildi!</span>
                </motion.div>
              )}

              {status === "error" && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-3"
                >
                  <FiAlertCircle className="w-5 h-5 flex-shrink-0" />
                  <span>Bir hata oluştu, lütfen tekrar deneyin</span>
                </motion.div>
              )}
            </div>
          </form>
        </div>
      </motion.main>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-8 mt-16">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div>
              <h3 className="text-white font-medium mb-4">İletişim</h3>
              <p className="text-sm">123 İş Merkezi</p>
              <p className="text-sm">İstanbul, Türkiye</p>
              <p className="text-sm mt-2">+90 555 123 45 67</p>
            </div>
            <div>
              <h3 className="text-white font-medium mb-4">Sosyal Medya</h3>
              <div className="flex justify-center space-x-4">
                <a href="#" className="hover:text-blue-400 transition-colors">
                  LinkedIn
                </a>
                <a href="#" className="hover:text-blue-400 transition-colors">
                  Twitter
                </a>
                <a href="#" className="hover:text-blue-400 transition-colors">
                  Instagram
                </a>
              </div>
            </div>
            <div>
              <h3 className="text-white font-medium mb-4">Çalışma Saatleri</h3>
              <p className="text-sm">Pazartesi - Cuma</p>
              <p className="text-sm">09:00 - 18:00</p>
            </div>
          </div>
          <div className="border-t border-gray-800 pt-8 text-sm">
            <p>&copy; {new Date().getFullYear()} Ürün Analizi. Tüm hakları saklıdır.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}