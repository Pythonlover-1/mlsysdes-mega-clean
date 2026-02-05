import { useState, useEffect } from 'react';

export function useImageCrop(
  imageDataUrl: string | undefined,
  box: number[] | undefined
): string | null {
  const [croppedUrl, setCroppedUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!imageDataUrl || !box || box.length < 4) {
      setCroppedUrl(null);
      return;
    }

    const [x1, y1, x2, y2] = box;
    const width = x2 - x1;
    const height = y2 - y1;

    if (width <= 0 || height <= 0) {
      setCroppedUrl(null);
      return;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        setCroppedUrl(null);
        return;
      }

      // Add some padding around the crop
      const padding = 10;
      const cropX = Math.max(0, x1 - padding);
      const cropY = Math.max(0, y1 - padding);
      const cropWidth = Math.min(img.width - cropX, width + padding * 2);
      const cropHeight = Math.min(img.height - cropY, height + padding * 2);

      canvas.width = cropWidth;
      canvas.height = cropHeight;

      ctx.drawImage(
        img,
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        0,
        0,
        cropWidth,
        cropHeight
      );

      setCroppedUrl(canvas.toDataURL('image/png'));
    };

    img.onerror = () => {
      setCroppedUrl(null);
    };

    img.src = imageDataUrl;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageDataUrl, box]);

  return croppedUrl;
}
