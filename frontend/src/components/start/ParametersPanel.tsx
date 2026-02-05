import { useState } from 'react';
import { useAppStore } from '../../store';
import { DEFAULT_CONVERSION_PARAMS } from '../../types';

function Slider({
  label,
  description,
  value,
  onChange,
  min,
  max,
  step,
  displayValue,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  displayValue?: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm text-white font-medium">{label}</label>
        <span className="text-sm text-cyan-400 font-mono">{displayValue ?? value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
      />
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );
}

function Toggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-start gap-3">
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-black ${
          checked ? 'bg-cyan-500' : 'bg-gray-600'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
      <div className="flex-1">
        <p className="text-sm text-white font-medium">{label}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
    </div>
  );
}

function TextInput({
  label,
  description,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  description: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm text-white font-medium block">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
      />
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );
}

function Section({
  title,
  icon,
  children,
  defaultOpen = false,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-800/50 hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && <div className="p-4 space-y-4 bg-gray-900/30">{children}</div>}
    </div>
  );
}

export default function ParametersPanel() {
  const params = useAppStore((s) => s.conversionParams);
  const setParams = useAppStore((s) => s.setConversionParams);
  const resetParams = useAppStore((s) => s.resetConversionParams);

  const isModified = JSON.stringify(params) !== JSON.stringify(DEFAULT_CONVERSION_PARAMS);

  return (
    <div className="p-6 bg-black rounded-2xl border border-gray-700 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center">
            <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
              />
            </svg>
          </div>
          <h2 className="font-bold text-white text-lg">Параметры анализа</h2>
        </div>
        {isModified && (
          <button
            onClick={resetParams}
            className="px-3 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            Сбросить
          </button>
        )}
      </div>

      <div className="space-y-3">
        {/* Detection Section */}
        <Section
          title="Детекция"
          defaultOpen={true}
          icon={
            <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          }
        >
          <Slider
            label="Порог уверенности"
            description="Минимальная уверенность для обнаружения элементов"
            value={params.detection_threshold}
            onChange={(v) => setParams({ detection_threshold: v })}
            min={0.1}
            max={0.95}
            step={0.05}
            displayValue={`${Math.round(params.detection_threshold * 100)}%`}
          />
          <Slider
            label="Утолщение линий"
            description="Расширить линии перед детекцией стрелок (0 = выкл)"
            value={params.arrow_dilate}
            onChange={(v) => setParams({ arrow_dilate: v })}
            min={0}
            max={10}
            step={1}
            displayValue={params.arrow_dilate === 0 ? 'выкл' : `${params.arrow_dilate}px`}
          />
        </Section>

        {/* Graph Building Section */}
        <Section
          title="Построение графа"
          icon={
            <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
        >
          <Slider
            label="Порог соединения"
            description="Максимальное расстояние для соединения стрелки с узлом (px)"
            value={params.connection_threshold}
            onChange={(v) => setParams({ connection_threshold: v })}
            min={50}
            max={500}
            step={10}
            displayValue={`${params.connection_threshold}px`}
          />
          <Slider
            label="IoU перекрытия"
            description="Порог IoU для подавления перекрывающихся объектов"
            value={params.overlap_iou}
            onChange={(v) => setParams({ overlap_iou: v })}
            min={0.3}
            max={0.95}
            step={0.05}
            displayValue={`${Math.round(params.overlap_iou * 100)}%`}
          />
          <Toggle
            label="Fallback для стрелок"
            description="Использовать bbox когда keypoints не работают"
            checked={params.arrow_fallback}
            onChange={(v) => setParams({ arrow_fallback: v })}
          />
          <Toggle
            label="Простой режим"
            description="Для диаграмм без pools/lanes — упрощённая обработка"
            checked={params.simple_mode}
            onChange={(v) => setParams({ simple_mode: v })}
          />
          {params.simple_mode && (
            <>
              <Slider
                label="Pool → Task ratio"
                description="Соотношение размера для превращения pool в task"
                value={params.simple_pool_task_size_ratio}
                onChange={(v) => setParams({ simple_pool_task_size_ratio: v })}
                min={0.3}
                max={1.0}
                step={0.05}
                displayValue={`${Math.round(params.simple_pool_task_size_ratio * 100)}%`}
              />
              <Slider
                label="IoU слияния блоков"
                description="Порог IoU для слияния перекрывающихся блоков"
                value={params.simple_merge_iou}
                onChange={(v) => setParams({ simple_merge_iou: v })}
                min={0.3}
                max={0.95}
                step={0.05}
                displayValue={`${Math.round(params.simple_merge_iou * 100)}%`}
              />
            </>
          )}
        </Section>

        {/* OCR Preprocessing Section */}
        <Section
          title="OCR: Предобработка"
          icon={
            <svg className="w-4 h-4 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          }
        >
          <Slider
            label="Увеличение"
            description="Масштабирование изображения перед OCR"
            value={params.ocr_upscale}
            onChange={(v) => setParams({ ocr_upscale: v })}
            min={1}
            max={4}
            step={1}
            displayValue={`${params.ocr_upscale}x`}
          />
          <Toggle
            label="Бинаризация"
            description="Преобразовать в черно-белое перед OCR"
            checked={params.ocr_binarize}
            onChange={(v) => setParams({ ocr_binarize: v })}
          />
          {params.ocr_binarize && (
            <Slider
              label="Порог бинаризации"
              description="Пороговое значение для преобразования в ч/б"
              value={params.ocr_binarize_thr}
              onChange={(v) => setParams({ ocr_binarize_thr: v })}
              min={50}
              max={250}
              step={5}
            />
          )}
          <Toggle
            label="Шумоподавление"
            description="Применить медианный фильтр для уменьшения шума"
            checked={params.ocr_denoise}
            onChange={(v) => setParams({ ocr_denoise: v })}
          />
          <Toggle
            label="Авто-детект шума"
            description="Автоматически определять зашумлённые изображения"
            checked={params.ocr_noisy_mode}
            onChange={(v) => setParams({ ocr_noisy_mode: v })}
          />
          {params.ocr_noisy_mode && (
            <Slider
              label="Порог шума (std)"
              description="Порог стандартного отклонения для определения шума"
              value={params.ocr_noisy_std}
              onChange={(v) => setParams({ ocr_noisy_std: v })}
              min={10}
              max={80}
              step={5}
            />
          )}
        </Section>

        {/* OCR Crop Settings */}
        <Section
          title="OCR: Обрезка"
          icon={
            <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z" />
            </svg>
          }
        >
          <Slider
            label="Inset (px)"
            description="Отступ внутрь в пикселях"
            value={params.ocr_inset_px}
            onChange={(v) => setParams({ ocr_inset_px: v })}
            min={0}
            max={10}
            step={1}
            displayValue={`${params.ocr_inset_px}px`}
          />
          <Slider
            label="Inset (%)"
            description="Отступ внутрь в процентах"
            value={params.ocr_inset_pct}
            onChange={(v) => setParams({ ocr_inset_pct: v })}
            min={0}
            max={0.2}
            step={0.01}
            displayValue={`${Math.round(params.ocr_inset_pct * 100)}%`}
          />
          <Slider
            label="Расширение bbox"
            description="Во сколько раз расширить область для OCR (1.0 = без расширения)"
            value={params.ocr_loose_scale}
            onChange={(v) => setParams({ ocr_loose_scale: v })}
            min={1.0}
            max={2.0}
            step={0.05}
            displayValue={`${params.ocr_loose_scale.toFixed(2)}x`}
          />
        </Section>

        {/* OCR Tesseract Section */}
        <Section
          title="OCR: Tesseract"
          icon={
            <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          }
        >
          <TextInput
            label="Языки"
            description="Коды языков Tesseract (например: eng+rus)"
            value={params.ocr_lang}
            onChange={(v) => setParams({ ocr_lang: v })}
            placeholder="eng+rus"
          />
          <TextInput
            label="PSM режимы"
            description="Режимы сегментации страницы через запятую"
            value={params.ocr_psm_list}
            onChange={(v) => setParams({ ocr_psm_list: v })}
            placeholder="6,7"
          />
          <Slider
            label="OEM режим"
            description="OCR Engine Mode (0=Legacy, 1=LSTM, 2=Both, 3=Default)"
            value={params.ocr_oem}
            onChange={(v) => setParams({ ocr_oem: v })}
            min={0}
            max={3}
            step={1}
          />
          <TextInput
            label="Доп. конфиг"
            description="Дополнительные параметры Tesseract"
            value={params.ocr_tesseract_config}
            onChange={(v) => setParams({ ocr_tesseract_config: v })}
            placeholder="-c preserve_interword_spaces=1"
          />
          <Toggle
            label="Whitelist по языку"
            description="Использовать whitelist символов на основе языка"
            checked={params.ocr_tesseract_whitelist_lang}
            onChange={(v) => setParams({ ocr_tesseract_whitelist_lang: v })}
          />
        </Section>

        {/* OCR Postprocessing Section */}
        <Section
          title="OCR: Постобработка"
          icon={
            <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          }
        >
          <Toggle
            label="Исправление раскладки"
            description="Исправлять смешанные кириллические/латинские символы"
            checked={params.ocr_fix_typos}
            onChange={(v) => setParams({ ocr_fix_typos: v })}
          />
          <Toggle
            label="Проверка орфографии"
            description="Автоматическая коррекция слов по словарю"
            checked={params.ocr_spellcheck}
            onChange={(v) => setParams({ ocr_spellcheck: v })}
          />
          {params.ocr_spellcheck && (
            <>
              <Slider
                label="Порог совпадения"
                description="Минимальный процент совпадения для коррекции"
                value={params.ocr_spellcheck_score}
                onChange={(v) => setParams({ ocr_spellcheck_score: v })}
                min={70}
                max={100}
                step={1}
                displayValue={`${params.ocr_spellcheck_score}%`}
              />
              <Slider
                label="Размер словаря"
                description="Количество слов в словаре для проверки"
                value={params.ocr_spellcheck_topn}
                onChange={(v) => setParams({ ocr_spellcheck_topn: v })}
                min={5000}
                max={50000}
                step={1000}
              />
              <Slider
                label="Мин. частота слова"
                description="Минимальная частота слова (Zipf) для коррекции"
                value={params.ocr_spellcheck_min_zipf}
                onChange={(v) => setParams({ ocr_spellcheck_min_zipf: v })}
                min={1}
                max={6}
                step={0.5}
              />
            </>
          )}
          <TextInput
            label="Пользовательский словарь"
            description="Путь к файлу с дополнительными словами"
            value={params.ocr_user_vocab}
            onChange={(v) => setParams({ ocr_user_vocab: v })}
            placeholder="/path/to/vocab.txt"
          />
        </Section>
      </div>
    </div>
  );
}
