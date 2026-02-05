import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '../../store';

export default function UploadArea() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const stageFile = useAppStore((s) => s.stageFile);
  const stagedFile = useAppStore((s) => s.stagedFile);
  const clearStagedFile = useAppStore((s) => s.clearStagedFile);
  const isConverting = useAppStore((s) => s.isConverting);
  const startConversion = useAppStore((s) => s.startConversion);
  const conversionError = useAppStore((s) => s.conversionError);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (file && file.type.startsWith('image/')) {
        stageFile(file);
      }
    },
    [stageFile]
  );

  // If file is staged, show preview
  if (stagedFile) {
    return (
      <div className="space-y-4">
        <div className="relative rounded-2xl border-2 border-white/20 bg-black overflow-hidden">
          {/* Preview image */}
          <div className="relative aspect-video flex items-center justify-center bg-slate-900/50 p-4">
            <img
              src={stagedFile.dataUrl}
              alt="Preview"
              className="max-w-full max-h-[400px] object-contain rounded-lg"
            />
          </div>

          {/* Filename */}
          <div className="p-4 border-t border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">{stagedFile.name}</p>
                <p className="text-xs text-gray-500">Готово к обработке</p>
              </div>
            </div>
            <button
              onClick={clearStagedFile}
              disabled={isConverting}
              className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Error message */}
        {conversionError && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-300">{conversionError}</p>
          </div>
        )}

        {/* Start button */}
        <button
          onClick={startConversion}
          disabled={isConverting}
          className="w-full py-4 px-6 bg-white text-black font-bold text-lg rounded-xl hover:bg-gray-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
        >
          {isConverting ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              Обработка...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Начать обработку
            </>
          )}
        </button>
      </div>
    );
  }

  // Upload area
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFile(e.dataTransfer.files[0]);
      }}
      onClick={() => inputRef.current?.click()}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          inputRef.current?.click();
        }
      }}
      className={`relative border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 overflow-hidden group focus:outline-none focus:ring-2 focus:ring-gray-500
        ${dragOver ? 'border-white bg-white/10 scale-105' : 'border-gray-700 hover:border-white/50 bg-black'}`}
    >
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* Corner Decorations */}
      <div className="absolute top-0 left-0 w-16 h-16 border-t-4 border-l-4 border-white opacity-0 group-hover:opacity-100 transition-all duration-300 rounded-tl-2xl" />
      <div className="absolute top-0 right-0 w-16 h-16 border-t-4 border-r-4 border-white opacity-0 group-hover:opacity-100 transition-all duration-300 rounded-tr-2xl" />
      <div className="absolute bottom-0 left-0 w-16 h-16 border-b-4 border-l-4 border-white opacity-0 group-hover:opacity-100 transition-all duration-300 rounded-bl-2xl" />
      <div className="absolute bottom-0 right-0 w-16 h-16 border-b-4 border-r-4 border-white opacity-0 group-hover:opacity-100 transition-all duration-300 rounded-br-2xl" />

      <div className="relative z-10 space-y-6">
        {/* Icon */}
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-white/20 to-white/10 border-2 border-white/30 group-hover:scale-110 transition-all duration-300">
          <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>

        {/* Text */}
        <div className="space-y-2">
          <p className="text-xl font-semibold text-white">{dragOver ? 'Отпустите файл' : 'Загрузить диаграмму'}</p>
          <p className="text-sm text-gray-400">Перетащите изображение сюда или кликните для выбора</p>
        </div>

        {/* Supported Formats */}
        <div className="flex items-center justify-center gap-3 text-xs text-gray-500">
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700">PNG</span>
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700">JPG</span>
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700">WEBP</span>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
    </div>
  );
}
