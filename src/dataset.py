import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.vocab import VietnameseVocab
from src.preprocessor import ImagePreprocessor

class VietnameseHTRDataset(Dataset):
    def __init__(self, annotation_file: str, vocab: VietnameseVocab, preprocessor: ImagePreprocessor):
        self.samples = []
        self.vocab = vocab
        self.preprocessor = preprocessor

        # Đọc file danh sách nhãn
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                img_path, transcription = line.split("\t", 1)
                self.samples.append((img_path, transcription))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Xử lý ảnh thành Tensor
        img_tensor = self.preprocessor.process(img_path)
        
        # Encode nhãn văn bản thành chuỗi Token IDs
        tokens = self.vocab.encode(text, add_special_tokens=True)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return img_tensor, tokens_tensor, text

def htr_collate_fn(batch, pad_token_id: int):
    """
    Hàm gom nhóm batch dữ liệu:
    - Stack các tensor ảnh về Tensor: [Batch, 1, H, W]
    - Pad các chuỗi token nhãn về cùng độ dài lớn nhất trong batch: [Batch, Max_Seq_Len]
    """
    images, tokens_list, raw_texts = zip(*batch)

    # 1. Stack ảnh (do tất cả ảnh đã được pad chuẩn về kích thước 64x512)
    images_tensor = torch.stack(images, dim=0)

    # 2. Pad chuỗi nhãn văn bản bằng pad_token_id
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_token_id)

    return {
        "images": images_tensor,
        "tokens": padded_tokens,
        "raw_texts": raw_texts
    }

def create_dataloader(annotation_file: str, vocab: VietnameseVocab, 
                      batch_size: int = 16, shuffle: bool = True, num_workers: int = 2):
    preprocessor = ImagePreprocessor(target_height=64, target_width=512)
    dataset = VietnameseHTRDataset(annotation_file, vocab, preprocessor)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: htr_collate_fn(b, pad_token_id=vocab.pad_id),
        pin_memory=torch.cuda.is_available()
    )
    return loader