import json

class VietnameseVocab:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        # Bảng ký tự tiếng Việt chuẩn (Chữ thường + Chữ hoa)
        vietnamese_chars = (
            "aàảãáạăằẳẵắặâầẩẫấậ"
            "bcdđeèẻẽéẹêềểễếệ"
            "fghiìỉĩíịjklmnoòỏõóọ"
            "ôồổỗốộơờởỡớợpqrstuùủũúụ"
            "ưừửữứựvwxyỳỷỹýỵz"
        )
        # Thêm chữ in hoa
        vietnamese_chars += vietnamese_chars.upper()
        
        # Thêm chữ số và các dấu câu thông dụng
        digits = "0123456789"
        punctuation = " .,!?:;\"'()-/\\%*+[]"

        # Tập hợp danh sách ký tự duy nhất
        unique_chars = list(dict.fromkeys(vietnamese_chars + digits + punctuation))
        self.chars = self.special_tokens + unique_chars

        # Xây dựng bảng ánh xạ hai chiều
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

        self.pad_id = self.char2idx[self.PAD_TOKEN]
        self.sos_id = self.char2idx[self.SOS_TOKEN]
        self.eos_id = self.char2idx[self.EOS_TOKEN]
        self.unk_id = self.char2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.chars)

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Chuyển chuỗi văn bản thành chuỗi số nguyên (Token IDs)."""
        tokens = [self.char2idx.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            tokens = [self.sos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: list, remove_special_tokens: bool = True) -> str:
        """Chuyển chuỗi số nguyên ngược lại thành văn bản."""
        chars = []
        for idx in token_ids:
            if idx == self.pad_id and remove_special_tokens:
                continue
            if idx == self.eos_id and remove_special_tokens:
                break
            if idx in (self.sos_id, self.unk_id) and remove_special_tokens:
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save_vocab(self, file_path: str = "vocab.json"):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path: str = "vocab.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.chars = json.load(f)
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}