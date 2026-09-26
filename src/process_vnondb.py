import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def extract_label_from_tracegroup(tg):
    """Trích xuất nhãn văn bản tiếng Việt từ thẻ traceGroup."""
    # 1. Tìm trong thẻ annotationXML
    ann_xml = tg.find(".//{http://www.w3.org/2003/InkML}annotationXML")
    if ann_xml is None:
        ann_xml = tg.find(".//annotationXML")

    if ann_xml is not None:
        # itertext() lấy toàn bộ nội dung text kể cả các thẻ con lồng bên trong
        text = "".join(ann_xml.itertext()).strip()
        if text:
            return text

    # 2. Dự phòng: tìm trong các thẻ annotation thông thường
    for ann in tg.findall(".//{http://www.w3.org/2003/InkML}annotation") + tg.findall(".//annotation"):
        text = "".join(ann.itertext()).strip()
        if text:
            return text

    return ""

def parse_strokes_from_tracegroup(tg):
    """Trích xuất danh sách tọa độ các nét vẽ thuộc traceGroup của dòng."""
    strokes = []
    traces = tg.findall(".//{http://www.w3.org/2003/InkML}trace") + tg.findall(".//trace")

    for trace in traces:
        if not trace.text:
            continue
        coords_str = trace.text.strip().split(',')
        stroke = []
        for pt in coords_str:
            parts = pt.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    stroke.append((x, y))
                except ValueError:
                    continue
        if stroke:
            strokes.append(stroke)

    return strokes

def render_strokes_to_image(strokes, output_png_path, line_thickness=2, padding=10):
    """Vẽ chuỗi tọa độ nét bút lên ảnh trắng nền đen và lưu file .png."""
    if not strokes:
        return False

    all_x = [pt[0] for stroke in strokes for pt in stroke]
    all_y = [pt[1] for stroke in strokes for pt in stroke]

    if not all_x or not all_y:
        return False

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    w = int(max_x - min_x) + padding * 2
    h = int(max_y - min_y) + padding * 2

    if w <= 0 or h <= 0:
        return False

    # Nền trắng (255)
    canvas = np.full((max(h, 20), max(w, 20)), 255, dtype=np.uint8)

    # Nối các điểm bằng đoạn thẳng nét đen (0) có khử răng cưa
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            pt1 = (int(stroke[i][0] - min_x + padding), int(stroke[i][1] - min_y + padding))
            pt2 = (int(stroke[i+1][0] - min_x + padding), int(stroke[i+1][1] - min_y + padding))
            cv2.line(canvas, pt1, pt2, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    cv2.imwrite(output_png_path, canvas)
    return True

def process_vnondb_split(split_txt_file, ink_dir, output_img_dir, output_annotation_file):
    if not os.path.exists(split_txt_file):
        print(f"[BỎ QUA] Không tìm thấy danh sách: {split_txt_file}")
        return

    split_name = os.path.basename(split_txt_file)
    print(f"\n==================================================")
    print(f"      ĐANG XỬ LÝ TẬP: {split_name}")
    print(f"==================================================")

    with open(split_txt_file, "r", encoding="utf-8") as f:
        doc_files = [line.strip() for line in f if line.strip()]

    print(f"Tổng số văn bản tài liệu: {len(doc_files)}")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_annotation_file), exist_ok=True)

    total_lines_rendered = 0

    with open(output_annotation_file, "w", encoding="utf-8") as out_f:
        for doc_idx, doc_name in enumerate(doc_files, 1):
            inkml_path = os.path.join(ink_dir, doc_name)
            if not os.path.exists(inkml_path):
                continue

            try:
                tree = ET.parse(inkml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Lỗi đọc {doc_name}: {e}")
                continue

            # Lấy tất cả traceGroup (từng dòng chữ) trong tài liệu
            trace_groups = root.findall(".//{http://www.w3.org/2003/InkML}traceGroup") + root.findall(".//traceGroup")

            doc_stem = os.path.splitext(doc_name)[0]
            lines_in_doc = 0

            for line_idx, tg in enumerate(trace_groups):
                label = extract_label_from_tracegroup(tg)
                strokes = parse_strokes_from_tracegroup(tg)

                # Chỉ ghi nhận nếu có cả nhãn chữ và nét vẽ
                if label and strokes:
                    png_name = f"{doc_stem}_line_{line_idx:03d}.png"
                    png_path = os.path.join(output_img_dir, png_name)

                    if render_strokes_to_image(strokes, png_path):
                        clean_path = os.path.abspath(png_path).replace("\\", "/")
                        out_f.write(f"{clean_path}\t{label}\n")
                        lines_in_doc += 1
                        total_lines_rendered += 1

            if doc_idx % 20 == 0 or doc_idx == len(doc_files):
                print(f"-> Đã xử lý [{doc_idx}/{len(doc_files)}] văn bản | Tổng số dòng chữ đã xuất: {total_lines_rendered}")

    print(f"==> HOÀN TẤT: Xuất thành công {total_lines_rendered} dòng chữ vào file: {output_annotation_file}")

def main():
    BASE_VNONDB_DIR = r"D:/KhoaLuanTotNghiep/VNOnDB_Line"
    INK_DIR = os.path.join(BASE_VNONDB_DIR, "InkData_line")

    OUTPUT_IMAGE_DIR = "data/processed/VNOnDB_Line/images"
    ANNOTATION_DIR = "data"

    # 1. Xử lý tập Train
    process_vnondb_split(
        split_txt_file=os.path.join(BASE_VNONDB_DIR, "train_set.txt"),
        ink_dir=INK_DIR,
        output_img_dir=os.path.join(OUTPUT_IMAGE_DIR, "train"),
        output_annotation_file=os.path.join(ANNOTATION_DIR, "vnondb_train_annotations.txt")
    )

    # 2. Xử lý tập Validation
    process_vnondb_split(
        split_txt_file=os.path.join(BASE_VNONDB_DIR, "validation_set.txt"),
        ink_dir=INK_DIR,
        output_img_dir=os.path.join(OUTPUT_IMAGE_DIR, "val"),
        output_annotation_file=os.path.join(ANNOTATION_DIR, "vnondb_val_annotations.txt")
    )

    # 3. Xử lý tập Test
    process_vnondb_split(
        split_txt_file=os.path.join(BASE_VNONDB_DIR, "test_set.txt"),
        ink_dir=INK_DIR,
        output_img_dir=os.path.join(OUTPUT_IMAGE_DIR, "test"),
        output_annotation_file=os.path.join(ANNOTATION_DIR, "vnondb_test_annotations.txt")
    )

if __name__ == "__main__":
    main()