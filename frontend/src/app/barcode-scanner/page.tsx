"use client";

import { useState, useEffect, useRef } from "react";
import { BrowserMultiFormatReader } from "@zxing/browser";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

type ProductData = {
  product_name: string;
  ingredients_text: string;
};

type AnalysisResult = {
  harmful_ingredients: string[];
  safe_ingredients: string[];
  toxicity_level: "Low" | "Medium" | "High";
  models_used?: string[];
};

// ✅ SmartLabel barkod çekme fonksiyonu
const fetchSmartLabelBarcode = async (url: string): Promise<string | null> => {
  try {
    const response = await fetch(url);
    const text = await response.text();
    const barcodeMatch = text.match(/\b(UPC|EAN)[\s:]+(\d{8,14})\b/i);
    return barcodeMatch ? barcodeMatch[2] : null;
  } catch (error) {
    console.error("SmartLabel barkod çekilemedi:", error);
    return null;
  }
};

export default function BarcodeScanner() {
  const router = useRouter();
  const [barcode, setBarcode] = useState<string | null>(null);
  const [product, setProduct] = useState<ProductData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const readerRef = useRef<BrowserMultiFormatReader | null>(null);

  const startScanning = async () => {
    try {
      setScanning(true);
      setError(null);
      setAnalysis(null);
      setProduct(null);

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" } 
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      readerRef.current = new BrowserMultiFormatReader();
      
      // DÜZELTME BURADA
      readerRef.current.decodeFromVideoDevice(
        undefined,
        videoRef.current!, // Non-null assertion
        (result, err) => {
          if (result) {
            stopScanning();
            handleScannedCode(result.getText());
          }
          if (err) console.error("Taramada hata:", err);
        }
      );
    } catch (err: unknown) {
      stopScanning();
      setError(err instanceof Error ? err.message : "Kamera erişimi reddedildi");
    }
  };

  const stopScanning = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    setScanning(false);
  };

  const handleScannedCode = async (code: string) => {
    setBarcode(code);
    setError(null);
    setLoading(true);

    try {
      let productData: ProductData | null = null;

      // Barkod/QR kod işleme
      if (/^\d{8,14}$/.test(code)) {
        const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${code}.json`);
        const data = await response.json();
        if (data.status === 1) productData = data.product;
      } else if (code.startsWith("http")) {
        const extractedBarcode = await fetchSmartLabelBarcode(code);
        if (extractedBarcode) {
          const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${extractedBarcode}.json`);
          const data = await response.json();
          if (data.status === 1) productData = data.product;
        }
      }

      if (!productData?.ingredients_text) {
        throw new Error("Ürün içerik bilgisi bulunamadı");
      }

      setProduct(productData);

      // AI analizi
      const analysisResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ingredients: productData.ingredients_text }),
      });

      if (!analysisResponse.ok) throw new Error("Analiz başarısız oldu");
      
      const analysisData = await analysisResponse.json();
      if (analysisData.status === "success") {
        setAnalysis(analysisData.analysis);
      } else {
        throw new Error(analysisData.message);
      }

    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Beklenmeyen bir hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisResults = () => {
    if (!analysis) return null;

    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="mt-4 p-4 border rounded-lg bg-gray-50 shadow"
      >
        <h3 className="text-lg font-bold text-blue-600 mb-3">Analiz Sonuçları</h3>

        {/* Zararlı Bileşenler */}
        {analysis.harmful_ingredients.length > 0 && (
          <div className="mb-3">
            <h4 className="text-red-600 font-semibold">❌ Zararlı Bileşenler</h4>
            {analysis.harmful_ingredients.map((ingredient, index) => (
              <p key={index} className="text-sm text-red-500">{ingredient}</p>
            ))}
          </div>
        )}

        {/* Güvenli Bileşenler */}
        {analysis.safe_ingredients.length > 0 && (
          <div className="mb-3">
            <h4 className="text-green-600 font-semibold">✅ Güvenli Bileşenler</h4>
            {analysis.safe_ingredients.map((ingredient, index) => (
              <p key={index} className="text-sm text-green-500">{ingredient}</p>
            ))}
          </div>
        )}

        {/* Toksisite Seviyesi */}
        <div className="mt-3">
          <h4 className="text-yellow-600 font-semibold">⚠️ Toksisite Seviyesi</h4>
          <p className={`text-sm ${
            analysis.toxicity_level === 'High' ? 'text-red-500' :
            analysis.toxicity_level === 'Medium' ? 'text-yellow-500' : 'text-green-500'
          }`}>
            {analysis.toxicity_level}
          </p>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="max-w-lg mx-auto bg-white p-6 rounded-lg shadow-md">
      <h1 className="text-2xl font-bold text-blue-600 text-center">📷 Barkod/QR Tarayıcı</h1>
      
      <div className="mt-4 flex justify-center">
        <video ref={videoRef} className="w-full max-w-md border rounded-md shadow-md" />
      </div>

      <div className="mt-4 space-y-4">
        {scanning && <p className="text-blue-500 text-center">📸 Tarama devam ediyor...</p>}
        {loading && <p className="text-gray-500 text-center">🔍 Analiz yapılıyor...</p>}
        
        {barcode && (
          <motion.p 
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="text-center text-green-600 font-medium"
          >
            Taranan Kod: {barcode}
          </motion.p>
        )}

        {error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-red-500 text-center"
          >
            {error}
          </motion.p>
        )}

        {product && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-4 border rounded bg-gray-50"
          >
            <h2 className="text-xl font-semibold mb-2">{product.product_name}</h2>
            <p className="text-gray-600">
              {product.ingredients_text || "İçerik bilgisi mevcut değil"}
            </p>
          </motion.div>
        )}

        {renderAnalysisResults()}

        <motion.button
          onClick={scanning ? stopScanning : startScanning}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="w-full bg-green-600 text-white py-3 rounded-lg shadow-lg hover:bg-green-700 transition"
        >
          {scanning ? "❌ Taramayı Durdur" : "📸 Taramayı Başlat"}
        </motion.button>
      </div>
    </div>
  );
}