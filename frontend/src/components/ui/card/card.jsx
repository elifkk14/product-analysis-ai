import React from "react";

// Card container
export function Card({ className = "", children, ...props }) {
  return (
    <div
      className={`rounded-lg bg-white shadow ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

// Card header alanı (genellikle başlık ve açıklama içerir)
export function CardHeader({ className = "", children, ...props }) {
  return (
    <div className={`border-b p-4 ${className}`} {...props}>
      {children}
    </div>
  );
}

// Card title bileşeni
export function CardTitle({ className = "", children, ...props }) {
  return (
    <h2 className={`text-xl font-semibold ${className}`} {...props}>
      {children}
    </h2>
  );
}

// Card description bileşeni
export function CardDescription({ className = "", children, ...props }) {
  return (
    <p className={`text-gray-600 ${className}`} {...props}>
      {children}
    </p>
  );
}

// Card content alanı
export function CardContent({ className = "", children, ...props }) {
  return (
    <div className={`p-4 ${className}`} {...props}>
      {children}
    </div>
  );
}
