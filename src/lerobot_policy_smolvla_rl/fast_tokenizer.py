import numpy as np
from scipy.fft import dct, idct
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class FastActionTokenizer:
    """Local implementation of the FAST action tokenizer based on the paper.

    Algorithm 1: FAST Tokenizer
    1. Compute DCT coefficients
    2. Quantize coefficients: round(gamma * C)
    3. Flatten tokens: Column-first (low-frequency components first)
    4. BPE Training / Tokenization
    """

    def __init__(
        self, scale: float = 10.0, vocab_size: int = 1024, bpe_path: str = None
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.tokenizer = None
        if bpe_path:
            self.tokenizer = Tokenizer.from_file(bpe_path)

    def encode_dct(self, actions: np.ndarray) -> np.ndarray:
        """Apply DCT and quantization.

        Args:
            actions: (batch, chunk_size, action_dim) in range [-1, 1]
        Returns:
            quantized_coeffs: (batch, chunk_size, action_dim) integers
        """
        # 1. DCT along the time axis (axis=1)
        # norm='ortho' matches physical-intelligence implementation
        coeffs = dct(actions, type=2, norm="ortho", axis=1)

        # 2. Quantize: round(gamma * C)
        quantized = np.round(self.scale * coeffs).astype(np.int32)
        return quantized

    def decode_dct(self, quantized: np.ndarray) -> np.ndarray:
        """Invert quantization and DCT."""
        coeffs = quantized.astype(np.float32) / self.scale
        actions = idct(coeffs, type=2, norm="ortho", axis=1)
        return actions

    def flatten(self, quantized: np.ndarray) -> list[list[int]]:
        """Flatten matrix column-first (interleaving dimensions).

        Args:
            quantized: (batch, H, D)
        Returns:
            list of lists of integers
        """
        batch_size, H, D = quantized.shape
        flattened = []
        for b in range(batch_size):
            # Column-first flattening: [C(1,1), C(2,1), ..., C(1,2), ..., C(D,H)]
            # Wait, the paper says: "interleaving action dimensions by including all
            # low-frequency components first".
            # This means: [C1_1, C2_1, ..., CD_1, C1_2, C2_2, ..., CD_2, ...]
            # which is (H, D) transposed and then flattened, or just axis=1 then axis=0.
            # In numpy, this is quantized[b].T.flatten()? No, that would be all frequencies of dim 1, then all of dim 2.
            # "all low-frequency components first" -> C1_1, C2_1, ..., CD_1 (all 1st freq)
            # then C1_2, C2_2, ..., CD_2 (all 2nd freq)
            # This is exactly quantized[b].flatten() if shape is (B, H, D).
            tokens = quantized[b].flatten().tolist()
            flattened.append(tokens)
        return flattened

    def unflatten(self, flattened: np.ndarray, H: int, D: int) -> np.ndarray:
        batch_size = flattened.shape[0]
        return flattened.reshape(batch_size, H, D)

    def fit(self, dataset_actions: np.ndarray):
        """Train BPE on a dataset of actions."""
        # 1. DCT and Flatten
        quantized = self.encode_dct(dataset_actions)
        flattened = self.flatten(quantized)

        # 2. Convert to strings for BPE
        # We map integers to characters to use standard BPE
        # To handle negative numbers, we shift them.
        self.min_val = np.min(quantized)
        self.max_val = np.max(quantized)

        text_data = []
        for seq in flattened:
            text_data.append(" ".join([str(x - self.min_val) for x in seq]))

        # 3. Train BPE
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]"]
        )
        self.tokenizer.train_from_iterator(text_data, trainer)

    def tokenize(self, actions: np.ndarray) -> list[list[int]]:
        """Full pipeline: DCT -> Flatten -> BPE."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")

        quantized = self.encode_dct(actions)
        flattened = self.flatten(quantized)

        token_ids = []
        for seq in flattened:
            text = " ".join([str(x - self.min_val) for x in seq])
            encoded = self.tokenizer.encode(text)
            token_ids.append(encoded.ids)

        return token_ids

    def detokenize(self, token_ids: list[list[int]], H: int, D: int) -> np.ndarray:
        """Invert BPE -> Unflatten -> IDCT."""
        quantized_flattened = []
        for ids in token_ids:
            decoded_text = self.tokenizer.decode(ids)
            seq = [int(x) + self.min_val for x in decoded_text.split()]
            quantized_flattened.append(seq)

        quantized = self.unflatten(np.array(quantized_flattened), H, D)
        return self.decode_dct(quantized)
