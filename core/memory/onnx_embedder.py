import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class FastONNXEmbedder:

  def __init__(
      self,
      model_path="all-MiniLM-L6-v2.onnx",
      tokenizer_path="tokenizer.json",
  ):
    # Quantized ONNX model load karein (Only ~45MB)
    self.session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.tokenizer.enable_padding(
        length=128, pad_id=0, pad_token="[PAD]"
    )
    self.tokenizer.enable_truncation(max_length=128)

  def encode(self, text: str) -> np.ndarray:
    encoded = self.tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    outputs = self.session.run(None, inputs)
    # Mean Pooling for Sentence Embeddings
    embeddings = outputs[0]  # shape: (1, seq_len, 384)
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    vector = sum_embeddings / sum_mask

    # Normalize vector for Cosine Similarity
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    return (vector / norm).astype("float32").flatten()
