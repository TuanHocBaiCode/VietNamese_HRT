import os
import torch
from src.vocab import VietnameseVocab
from src.dataset import create_dataloader

def test_dataset(annotation_path, dataset_name, vocab):
    print(f"   KIỂM TRA DỮ LIỆU: {dataset_name}")
    
    if not os.path.exists(annotation_path):
        print(f"Không tìm thấy file nhãn: {annotation_path}")
        return False

    # Đếm số dòng dữ liệu thực tế
    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"-> Tổng số mẫu ghi nhận: {len(lines)}")

    if len(lines) == 0:
        print("File nhãn bị rỗng!")
        return False

    # 1. Khởi tạo DataLoader với Batch Size = 4
    batch_size = 4
    try:
        loader = create_dataloader(
            annotation_file=annotation_path,
            vocab=vocab,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    except Exception as e:
        print(f"Khởi tạo DataLoader thất bại: {e}")
        return False

    # 2. Rút 1 batch đầu tiên để kiểm tra tensor
    try:
        batch = next(iter(loader))
        images = batch["images"]
        tokens = batch["tokens"]
        raw_texts = batch["raw_texts"]
    except Exception as e:
        print(f"Không thể đọc/xử lý batch ảnh đầu tiên: {e}")
        return False

    # 3. Kiểm tra kích thước Tensor
    print(f"-> Batch ảnh (images shape): {images.shape}")
    print(f"   (Kỳ vọng: torch.Size([{batch_size}, 1, 64, 512]))")
    
    print(f"-> Batch nhãn (tokens shape): {tokens.shape}")
    print(f"   (Kỳ vọng: torch.Size([{batch_size}, Max_Len_Trong_Batch]))")

    # 4. Kiểm tra khả năng Encode / Decode nhãn
    print("\n-> Kiểm tra đối chiếu Nhãn gốc vs Decode:")
    sample_raw = raw_texts[0]
    sample_decoded = vocab.decode(tokens[0].tolist(), remove_special_tokens=True)
    print(f"   [Gốc]:   {sample_raw}")
    print(f"   [Decode]: {sample_decoded}")

    if sample_raw == sample_decoded:
        print("-> Khớp 100% nhãn ký tự!")
    else:
        print("-> Có sự khác biệt nhỏ về ký tự (có thể do ký tự lạ chưa có trong từ vựng).")

    return True

def main():
    # 1. Khởi tạo bộ từ vựng chuẩn và lưu file vocab.json
    vocab = VietnameseVocab()
    vocab.save_vocab("vocab.json")
    print(f"Đã nạp bộ từ vựng: {len(vocab)} ký tự (lưu tại vocab.json)")

    # 2. Kiểm tra bộ dữ liệu UIT-HWDB
    test_dataset(
        annotation_path="data/train_annotations.txt",
        dataset_name="UIT-HWDB (Train)",
        vocab=vocab
    )

    # 3. Kiểm tra bộ dữ liệu VNOnDB
    test_dataset(
        annotation_path="data/vnondb_train_annotations.txt",
        dataset_name="VNOnDB_Line (Train)",
        vocab=vocab
    )

if __name__ == "__main__":
    main()