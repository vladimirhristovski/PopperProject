DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, text):
        return self._model.encode(text, normalize_embeddings=True).tolist()
