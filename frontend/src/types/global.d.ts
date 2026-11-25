// frontend/src/types/global.d.ts

export {};

declare global {
  interface Window {
    BarcodeDetector?: {
      new (options?: { formats: string[] }): BarcodeDetector;
    };
  }

  interface BarcodeDetector {
    detect(video: HTMLVideoElement): Promise<{ rawValue: string }[]>;
  }
}
