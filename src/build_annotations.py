import os
import json
from glob import glob

def process_split_folder(split_dir, output_file):
    """
    Quét qua các thư mục con (1, 2, 3...) trong thư mục train hoặc test,
    đọc file label.json trong từng thư mục và ghi ra file text nhãn.
    """
    if not os.path.exists(split_dir):
        print(f"Không tìm thấy thư mục: {split_dir}")
        return 0

    # Tìm tất cả các file label.json nằm trong các thư mục con
    json_pattern = os.path.join(split_dir, "**", "label.json")
    json_files = glob(json_pattern, recursive=True)

    if not json_files:
        print(f"Không tìm thấy file label.json nào trong: {split_dir}")
        return 0

    print(f"\n--- Đang xử lý: {split_dir} ---")
    print(f"Tìm thấy {len(json_files)} thư mục con có chứa file label.json.")

    # Đảm bảo thư mục lưu file txt đích đã tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_samples = 0
    missing_images = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for json_path in sorted(json_files):
            subfolder = os.path.dirname(json_path)

            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"Lỗi đọc file {json_path}: {e}")
                continue

            for img_name, label in data.items():
                img_path = os.path.join(subfolder, img_name)
                
                # Kiểm tra ảnh có thực sự tồn tại trong thư mục hay không
                if os.path.exists(img_path):
                    clean_label = label.strip()
                    if clean_label:
                        # Chuẩn hóa đường dẫn dấu gạch chéo xuôi '/'
                        normalized_path = os.path.abspath(img_path).replace("\\", "/")
                        out_f.write(f"{normalized_path}\t{clean_label}\n")
                        total_samples += 1
                else:
                    missing_images += 1

    print(f"-> Ghi thành công {total_samples} dòng vào: {output_file}")
    if missing_images > 0:
        print(f"-> Bỏ qua {missing_images} ảnh được liệt kê trong json nhưng không tồn tại trên đĩa.")
        
    return total_samples

def find_split_dir(base_dir, candidates):
    """Hỗ trợ tự động tìm kiếm thư mục theo các tên thông dụng (train/train_data...)"""
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None

def main():
    # 1. Đường dẫn thư mục gốc bộ dữ liệu UIT trên máy của bạn
    BASE_DIR = r"D:/KhoaLuanTotNghiep/UIT_HWDB_line"

    # 2. Tự động nhận diện thư mục train và test
    train_dir = find_split_dir(BASE_DIR, ["train_data", "train", "Train_data", "Train"])
    test_dir = find_split_dir(BASE_DIR, ["test_data", "test", "Test_data", "Test"])

    print(f"Thư mục gốc: {BASE_DIR}")
    print(f"Thư mục Train xác định được: {train_dir}")
    print(f"Thư mục Test xác định được:  {test_dir}")

    # 3. Xử lý tập Train
    if train_dir:
        train_output = "data/train_annotations.txt"
        process_split_folder(train_dir, train_output)
    else:
        print("\nKhông tìm thấy thư mục tập Train trong thư mục gốc!")

    # 4. Xử lý tập Test
    if test_dir:
        test_output = "data/test_annotations.txt"
        process_split_folder(test_dir, test_output)
    else:
        print("\nKhông tìm thấy thư mục tập Test trong thư mục gốc!")

if __name__ == "__main__":
    main()