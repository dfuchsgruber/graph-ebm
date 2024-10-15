from typing import List

import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from sentence_transformers import SentenceTransformer
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def transform_text_with_sentence_transformer(
    text: list[str], sentence_transformer: str
) -> Float[np.ndarray, "num_texts embedding_dim"]:
    """Uses a sentence transformer to encode a list of texts into a matrix of embeddings."""
    model = SentenceTransformer(sentence_transformer)
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        return model.encode(text)  # type: ignore
