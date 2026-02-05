import { useImageCrop } from '../../hooks/useImageCrop';

interface Props {
  imageDataUrl: string | undefined;
  box: number[] | undefined;
}

export default function CroppedImage({ imageDataUrl, box }: Props) {
  const croppedUrl = useImageCrop(imageDataUrl, box);

  if (!croppedUrl) {
    return (
      <div className="w-full h-32 bg-slate-800/50 rounded-lg flex items-center justify-center text-slate-600 text-sm">
        Изображение недоступно
      </div>
    );
  }

  return (
    <div className="w-full bg-slate-800/50 rounded-lg overflow-hidden border border-slate-700/50">
      <img
        src={croppedUrl}
        alt="Cropped element"
        className="w-full h-auto max-h-48 object-contain"
      />
    </div>
  );
}
