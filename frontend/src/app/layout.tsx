"use client";

import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { ThemeProvider, useTheme } from "next-themes";
import { useEffect, useState } from "react";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Updated color palette based on the reference images
const colors = {
  primary: '#B3B792',       // Soft green (from cosmetics site)
  primaryDark: '#809671',   // Darker green
  secondary: '#E5E0D8',     // Cream/beige background
  accent: '#D2AB80',        // Warm accent (like chai)
  textDark: '#725C3A',      // Dark brown text
  textLight: '#F8F5F0',     // Light text
  highlight: '#E5D2B8',     // Light highlight
  sale: '#FF6B6B',          // For sale/alert items
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [mounted, setMounted] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <html lang="tr" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#F8F5F0] dark:bg-[#2A2A2A] text-[#333333] dark:text-[#F8F5F0] min-h-screen flex flex-col`}
      >
        <ThemeProvider attribute="class">
          {mounted && (
            <>
              {/* Navbar */}
              <nav className={`bg-[#B3B792] dark:bg-[#1A1A1A] sticky top-0 z-50 shadow-sm`}>
                <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                  <div className="flex justify-between items-center h-16">
                    <div className="flex items-center">
                      <Link href="/" className="flex items-center">
                        <span className="text-2xl font-bold text-white mr-2">🧴</span>
                        <span className="text-xl font-bold text-white">ÜrünAnaliz</span>
                      </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-6">
                      <Link href="/" className="text-white hover:text-[#F8F5F0] font-medium transition px-2 py-1 rounded hover:bg-[#809671] hover:bg-opacity-30">
                        Anasayfa
                      </Link>
                      <Link href="/analysis" className="text-white hover:text-[#F8F5F0] font-medium transition px-2 py-1 rounded hover:bg-[#809671] hover:bg-opacity-30">
                        Analiz
                      </Link>
                      
                      <Link href="/about" className="text-white hover:text-[#F8F5F0] font-medium transition px-2 py-1 rounded hover:bg-[#809671] hover:bg-opacity-30">
                        Hakkımızda
                      </Link>
                      <Link href="/contact" className="text-white hover:text-[#F8F5F0] font-medium transition px-2 py-1 rounded hover:bg-[#809671] hover:bg-opacity-30">
                        İletişim
                      </Link>
                      <Link href="/faq" className="text-white hover:text-[#F8F5F0] font-medium transition px-2 py-1 rounded hover:bg-[#809671] hover:bg-opacity-30">
                        SSS
                      </Link>
                      
                      <div className="ml-2 bg-[#809671] dark:bg-[#333333] p-1 rounded-full flex items-center shadow-sm">
                        <button
                          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                          className="w-10 h-6 rounded-full bg-white dark:bg-[#444444] relative transition-colors duration-300 focus:outline-none"
                          aria-label="Toggle dark mode"
                        >
                          <div
                            className={`absolute top-1 w-4 h-4 rounded-full transition-all duration-300 transform flex items-center justify-center
                              ${theme === "dark" 
                                ? "translate-x-5 bg-[#D2AB80]" 
                                : "translate-x-1 bg-[#809671]"
                              }`}
                          >
                            {theme === "dark" ? "🌙" : "☀️"}
                          </div>
                        </button>
                      </div>
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="md:hidden flex items-center">
                      <button
                        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                        className="mr-4 bg-[#809671] dark:bg-[#333333] p-1 rounded-full flex items-center shadow-sm"
                        aria-label="Toggle dark mode"
                      >
                        <div className="w-8 h-5 rounded-full bg-white dark:bg-[#444444] relative transition-colors duration-300">
                          <div
                            className={`absolute top-0.5 w-4 h-4 rounded-full transition-all duration-300 transform flex items-center justify-center
                              ${theme === "dark" 
                                ? "translate-x-4 bg-[#D2AB80]" 
                                : "translate-x-0.5 bg-[#809671]"
                              }`}
                          >
                            {theme === "dark" ? "🌙" : "☀️"}
                          </div>
                        </div>
                      </button>

                      <button
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        className="text-white focus:outline-none"
                      >
                        <svg
                          className="h-6 w-6"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          {mobileMenuOpen ? (
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          ) : (
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M4 6h16M4 12h16M4 18h16"
                            />
                          )}
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Mobile Menu */}
                {mobileMenuOpen && (
                  <div className="md:hidden bg-[#B3B792] dark:bg-[#1A1A1A] shadow-sm">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                      <Link
                        href="/"
                        className="text-white hover:text-[#F8F5F0] block px-3 py-2 rounded-md font-medium hover:bg-[#809671] hover:bg-opacity-30"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        Anasayfa
                      </Link>
                      <Link
                        href="/analysis"
                        className="text-white hover:text-[#F8F5F0] block px-3 py-2 rounded-md font-medium hover:bg-[#809671] hover:bg-opacity-30"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        Analiz
                      </Link>
                      <Link
                        href="/about"
                        className="text-white hover:text-[#F8F5F0] block px-3 py-2 rounded-md font-medium hover:bg-[#809671] hover:bg-opacity-30"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        Hakkımızda
                      </Link>
                      <Link
                        href="/contact"
                        className="text-white hover:text-[#F8F5F0] block px-3 py-2 rounded-md font-medium hover:bg-[#809671] hover:bg-opacity-30"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        İletişim
                      </Link>
                      <Link
                        href="/faq"
                        className="text-white hover:text-[#F8F5F0] block px-3 py-2 rounded-md font-medium hover:bg-[#809671] hover:bg-opacity-30"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        SSS
                      </Link>
                    </div>
                  </div>
                )}
              </nav>

              {/* Page Content */}
              <main className="flex-grow">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                  {children}
                </div>
              </main>

              {/* Footer */}
              <footer className="bg-[#E5E0D8] dark:bg-[#1A1A1A] border-t border-[#D2AB80] dark:border-[#333333]">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                    <div>
                      <h3 className="text-lg font-semibold mb-4 text-[#725C3A] dark:text-[#F8F5F0]">ÜrünAnaliz</h3>
                      <p className="text-sm text-[#725C3A] dark:text-[#D2AB80]">
                        Ürün bileşenlerini analiz ederek sağlığınız için daha bilinçli kararlar vermenizi sağlıyoruz.
                      </p>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-4 text-[#725C3A] dark:text-[#F8F5F0]">Keşfet</h3>
                      <ul className="space-y-2 text-sm">
                        <li>
                          <Link href="/analysis" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            Ürün Analizi
                          </Link>
                        </li>
                        <li>
                          <Link href="/ingredients" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            Zararlı İçerikler
                          </Link>
                        </li>
                        <li>
                          <Link href="/categories" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            Ürün Kategorileri
                          </Link>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-4 text-[#725C3A] dark:text-[#F8F5F0]">Destek</h3>
                      <ul className="space-y-2 text-sm">
                        <li>
                          <Link href="/faq" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            Sık Sorulan Sorular
                          </Link>
                        </li>
                        <li>
                          <Link href="/contact" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            İletişim
                          </Link>
                        </li>
                        <li>
                          <Link href="/privacy" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                            Gizlilik Politikası
                          </Link>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-4 text-[#725C3A] dark:text-[#F8F5F0]">Bize Katılın</h3>
                      <div className="flex space-x-4">
                        <a href="#" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                          <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z" />
                          </svg>
                        </a>
                        <a href="#" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                          <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
                          </svg>
                        </a>
                        <a href="#" className="text-[#725C3A] dark:text-[#D2AB80] hover:text-[#809671] dark:hover:text-[#F8F5F0]">
                          <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M22.675 0h-21.35c-.732 0-1.325.593-1.325 1.325v21.351c0 .731.593 1.324 1.325 1.324h11.495v-9.294h-3.128v-3.622h3.128v-2.671c0-3.1 1.893-4.788 4.659-4.788 1.325 0 2.463.099 2.795.143v3.24l-1.918.001c-1.504 0-1.795.715-1.795 1.763v2.313h3.587l-.467 3.622h-3.12v9.293h6.116c.73 0 1.323-.593 1.323-1.325v-21.35c0-.732-.593-1.325-1.325-1.325z" />
                          </svg>
                        </a>
                      </div>
                      <div className="mt-4">
                        <form className="flex">
                          <input
                            type="email"
                            placeholder="E-posta adresiniz"
                            className="px-4 py-2 w-full rounded-l-md border border-[#D2AB80] dark:border-[#444444] bg-white dark:bg-[#2A2A2A] text-[#725C3A] dark:text-[#F8F5F0] focus:outline-none focus:ring-2 focus:ring-[#B3B792] text-sm"
                          />
                          <button
                            type="submit"
                            className="bg-[#B3B792] text-white px-4 py-2 rounded-r-md hover:bg-[#809671] focus:outline-none focus:ring-2 focus:ring-[#B3B792] text-sm"
                          >
                            Abone Ol
                          </button>
                        </form>
                      </div>
                    </div>
                  </div>
                  <div className="mt-8 pt-8 border-t border-[#D2AB80] dark:border-[#444444] text-center text-sm text-[#725C3A] dark:text-[#D2AB80]">
                    <p>© {new Date().getFullYear()} ÜrünAnaliz. Tüm hakları saklıdır.</p>
                  </div>
                </div>
              </footer>
            </>
          )}
        </ThemeProvider>
      </body>
    </html>
  );
}