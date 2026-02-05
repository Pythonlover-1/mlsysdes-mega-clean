"""
Пакетная обработка BPMN диаграмм с полной визуализацией этапов

Для каждого изображения создается отдельная папка со всеми результатами:
- detections.json - сырые результаты детекции
- detected_image.png - визуализация обнаруженных элементов
- graph.json - граф (узлы и рёбра)
- graph.csv - список рёбер в CSV
- graph.dot - граф для Graphviz
- ocr_tesseract.png - визуализация OCR (Tesseract)
- ocr_cv.png - визуализация текстовых областей (CV метод)
"""

import json
from pathlib import Path
from typing import List, Dict
from PIL import Image

# Import detection and graph conversion
from bpmn_detection_demo import (
    load_models, 
    detect_objects, 
    detect_arrows,
    visualize_detections,
    print_detection_summary
)
from bpmn_to_graph import detections_to_graph
from bpmn_ocr_visualize import (
    Box,
    _object_boxes_from_detector,
    _tesseract_boxes_in_box,
    _cv_text_regions_in_box,
    _draw_tesseract,
    _draw_cv
)


def process_single_image(
    image_path: Path,
    output_base: Path,
    model_object,
    model_arrow,
    device,
    threshold: float = 0.5,
    connection_threshold: float = 200
):
    """
    Обработка одного изображения со всеми этапами визуализации
    
    Args:
        image_path: Путь к изображению
        output_base: Базовая папка для результатов
        model_object: Модель детекции объектов
        model_arrow: Модель детекции стрелок
        device: Устройство PyTorch
        threshold: Порог уверенности для детекции
        connection_threshold: Максимальное расстояние для соединения стрелок
    """
    print(f"\n{'='*80}")
    print(f"Обработка: {image_path.name}")
    print(f"{'='*80}")
    
    # Создать папку для результатов
    output_dir = output_base / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузить изображение
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    print(f"  Размер изображения: {image_size[0]}x{image_size[1]}")
    
    # ========================================
    # Этап 1: Детекция элементов
    # ========================================
    print("\n[Этап 1] Детекция BPMN элементов")
    print("  → Детекция объектов (tasks, gateways, events, etc.)...")
    object_detections = detect_objects(image, model_object, device, threshold)
    
    print("  → Детекция стрелок (flows) с keypoints...")
    arrow_detections = detect_arrows(image, model_arrow, device, threshold)
    
    # Объединить все детекции
    all_detections = object_detections + arrow_detections
    
    # Сохранить сырые результаты детекции
    detections_path = output_dir / "detections.json"
    with open(detections_path, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Сохранён JSON детекций: {detections_path.name}")
    
    # Вывести статистику
    print_detection_summary(all_detections)
    
    # Визуализировать детекции
    detected_img_path = output_dir / "detected_image.png"
    visualize_detections(image.copy(), all_detections, str(detected_img_path))
    print(f"  ✓ Сохранена визуализация: {detected_img_path.name}")
    
    # ========================================
    # Этап 2: Преобразование в граф
    # ========================================
    print("\n[Этап 2] Преобразование в граф")
    graph = detections_to_graph(
        all_detections,
        connection_threshold=connection_threshold,
        image=image
    )
    graph.image_size = image_size
    
    # Статистика графа
    connected_edges = sum(1 for e in graph.edges if e.source_id and e.target_id)
    nodes_in_lanes = sum(1 for n in graph.nodes.values() if n.lane_id)
    print(f"  Узлов: {len(graph.nodes)}")
    print(f"  Рёбер: {len(graph.edges)} ({connected_edges} соединённых)")
    print(f"  Lanes/Pools: {len(graph.lanes)}")
    print(f"  Узлов в lanes: {nodes_in_lanes}/{len(graph.nodes)}")
    
    # Сохранить граф в разных форматах
    # JSON
    graph_json_path = output_dir / "graph.json"
    with open(graph_json_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_json())
    print(f"  ✓ Сохранён граф (JSON): {graph_json_path.name}")
    
    # CSV
    graph_csv_path = output_dir / "graph.csv"
    with open(graph_csv_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_csv())
    print(f"  ✓ Сохранён граф (CSV): {graph_csv_path.name}")
    
    # DOT
    graph_dot_path = output_dir / "graph.dot"
    with open(graph_dot_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_dot())
    print(f"  ✓ Сохранён граф (DOT): {graph_dot_path.name}")
    
    # ========================================
    # Этап 3: OCR визуализация
    # ========================================
    print("\n[Этап 3] OCR визуализация")
    
    # Получить bounding boxes объектов
    obj_boxes = _object_boxes_from_detector(image, model_object, device, threshold=threshold)
    print(f"  → Обработка {len(obj_boxes)} объектов для OCR...")
    
    # Tesseract OCR
    tesseract_boxes = []
    cv_boxes = []
    lang = "eng+rus"
    psm = "6"
    
    for region in obj_boxes:
        tesseract_boxes.extend(_tesseract_boxes_in_box(image, region, lang=lang, psm=psm))
        cv_boxes.extend(_cv_text_regions_in_box(image, region))
    
    # Визуализация Tesseract
    if tesseract_boxes:
        tesseract_vis = _draw_tesseract(image.copy(), tesseract_boxes)
        ocr_tess_path = output_dir / "ocr_tesseract.png"
        tesseract_vis.save(ocr_tess_path)
        print(f"  ✓ Сохранена OCR визуализация (Tesseract): {ocr_tess_path.name}")
        print(f"    Обнаружено текстовых блоков: {len(tesseract_boxes)}")
    else:
        print("  ⚠ Tesseract не обнаружил текста")
    
    # Визуализация CV
    cv_vis = _draw_cv(image.copy(), cv_boxes)
    ocr_cv_path = output_dir / "ocr_cv.png"
    cv_vis.save(ocr_cv_path)
    print(f"  ✓ Сохранена OCR визуализация (CV метод): {ocr_cv_path.name}")
    print(f"    Обнаружено текстовых областей: {len(cv_boxes)}")
    
    # ========================================
    # Сводка
    # ========================================
    print(f"\n{'='*80}")
    print(f"✓ Обработка завершена: {image_path.name}")
    print(f"  Результаты сохранены в: {output_dir}/")
    print(f"  Файлы:")
    print(f"    - detections.json       : Сырые результаты детекции")
    print(f"    - detected_image.png    : Визуализация обнаруженных элементов")
    print(f"    - graph.json            : Граф (узлы и рёбра)")
    print(f"    - graph.csv             : Список рёбер в CSV")
    print(f"    - graph.dot             : Граф для Graphviz")
    if tesseract_boxes:
        print(f"    - ocr_tesseract.png     : OCR визуализация (Tesseract)")
    print(f"    - ocr_cv.png            : Визуализация текстовых областей")
    print(f"{'='*80}")


def main():
    """Пакетная обработка BPMN диаграмм"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Пакетная обработка BPMN диаграмм с визуализацией этапов"
    )
    parser.add_argument(
        "--input-dir",
        default="data/Picture",
        help="Папка с изображениями"
    )
    parser.add_argument(
        "--output-dir",
        default="bpmn_batch_results",
        help="Папка для результатов"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Количество изображений для обработки"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог уверенности для детекции"
    )
    parser.add_argument(
        "--connection-threshold",
        type=float,
        default=200,
        help="Максимальное расстояние для соединения стрелок"
    )
    
    args = parser.parse_args()
    
    # Найти изображения
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Получить список изображений (PNG и JPG)
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    
    if not image_files:
        print(f"❌ Изображения не найдены в {input_dir}")
        return
    
    # Ограничить количество
    image_files = image_files[:args.count]
    
    print("\n" + "="*80)
    print("Пакетная обработка BPMN диаграмм")
    print("="*80)
    print(f"Входная папка: {input_dir}")
    print(f"Выходная папка: {output_base}")
    print(f"Изображений для обработки: {len(image_files)}")
    print(f"Порог детекции: {args.threshold}")
    print(f"Порог соединения: {args.connection_threshold}")
    print("="*80)
    
    # Загрузить модели (один раз)
    print("\nЗагрузка моделей...")
    model_object, model_arrow, device = load_models()
    print("✓ Модели загружены")
    
    # Обработать каждое изображение
    for i, image_path in enumerate(image_files, 1):
        print(f"\n\n[{i}/{len(image_files)}]")
        process_single_image(
            image_path,
            output_base,
            model_object,
            model_arrow,
            device,
            threshold=args.threshold,
            connection_threshold=args.connection_threshold
        )
    
    # Финальная сводка
    print("\n\n" + "="*80)
    print("🎉 ВСЕ ИЗОБРАЖЕНИЯ ОБРАБОТАНЫ!")
    print("="*80)
    print(f"Всего обработано: {len(image_files)} изображений")
    print(f"Результаты сохранены в: {output_base}/")
    print("\nСтруктура результатов:")
    print(f"  {output_base}/")
    for img in image_files[:3]:  # Показать первые 3
        print(f"    ├── {img.stem}/")
        print(f"    │   ├── detections.json")
        print(f"    │   ├── detected_image.png")
        print(f"    │   ├── graph.json")
        print(f"    │   ├── graph.csv")
        print(f"    │   ├── graph.dot")
        print(f"    │   ├── ocr_tesseract.png")
        print(f"    │   └── ocr_cv.png")
    if len(image_files) > 3:
        print(f"    └── ... ещё {len(image_files) - 3} папок")
    print("="*80)


if __name__ == "__main__":
    main()
