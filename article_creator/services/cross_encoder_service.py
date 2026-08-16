import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from radlab_data.text.processors.splitter import SentenceSplitter
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)


def split_to_fixed_chunks(
    text: str,
    tokenizer: Any,
    chunk_size_tokens: int,
    overlap_window_perc: float = 0.2,
) -> List[str]:
    """
    Split text into overlapping token-level chunks (sliding window).

    Each chunk covers chunk_size_tokens tokens; the window advances by
    (1 - overlap_window_perc) of that size.
    """
    if chunk_size_tokens <= 0 or not text:
        return []

    all_tokens = tokenizer.encode(text)
    n = len(all_tokens)
    if n <= chunk_size_tokens:
        return []

    step = max(1, chunk_size_tokens - int(chunk_size_tokens * overlap_window_perc))
    chunks: List[str] = []
    covered_end = 0

    for start in range(0, n - chunk_size_tokens + 1, step):
        end = start + chunk_size_tokens
        chunk_str = tokenizer.decode(all_tokens[start:end])
        if chunk_str.strip():
            chunks.append(chunk_str.strip())
        covered_end = end

    if covered_end < n:
        tail = tokenizer.decode(all_tokens[-chunk_size_tokens:])
        if tail.strip() and (not chunks or chunks[-1] != tail.strip()):
            chunks.append(tail.strip())

    return chunks


class CrossEncoderService:
    """
    Service encapsulating CrossEncoder model inference and similarity calculations.
    Thread-safe for concurrent access from multiple requests.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_instance: Optional[CrossEncoder] = None,
    ):
        if model_instance is not None:
            self.model = model_instance
            self.model_name = model_name or "custom-instance"
        else:
            self.model_name = model_name or os.environ.get(
                "CROSS_ENCODER_MODEL", "radlab/polish-cross-encoder"
            )
            device = device or os.environ.get("CROSS_ENCODER_DEVICE", "auto")
            logger.info(
                f"Loading CrossEncoder model '{self.model_name}' on device: '{device}'..."
            )
            self.model = CrossEncoder(self.model_name, device=device)
            logger.info(
                f"CrossEncoder model '{self.model_name}' loaded successfully."
            )

        self.splitter = SentenceSplitter()
        self._lock = threading.Lock()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def _calculate_similarity_patch_match(
        self,
        gen_text: str,
        orig_text: str,
        ce_max_tokens: int,
        special_tokens: int = 2,
    ) -> Optional[float]:
        """
        Calculates similarity using patch matching when texts exceed max token limit.
        """
        gen_sentences = [s for s in self.splitter.split(gen_text) if s and s.strip()]
        if not gen_sentences:
            return None

        tokenizer = self.tokenizer
        gen_token_lengths = [len(tokenizer.encode(s)) for s in gen_sentences]
        max_gen_tokens = max(gen_token_lengths)

        chunk_size = ce_max_tokens - max_gen_tokens - special_tokens
        if chunk_size <= 0:
            logger.warning(
                f"Patch matching fallback: longest gen sentence ({max_gen_tokens} tokens) "
                f"exceeds CE token limit {ce_max_tokens} (budget = {chunk_size})."
            )
            return None

        orig_chunks = split_to_fixed_chunks(
            orig_text, tokenizer, chunk_size, overlap_window_perc=0.2
        )
        if not orig_chunks:
            logger.warning(
                f"Patch matching fallback: orig_text could not form any chunk "
                f"(budget {chunk_size} tokens too small)."
            )
            return None

        valid_pairs: List[Tuple[int, int]] = []
        for gi in range(len(gen_sentences)):
            for ci in range(len(orig_chunks)):
                valid_pairs.append((gi, ci))

        if not valid_pairs:
            logger.warning(
                f"Patch matching fallback: no (gen_sentence, orig_chunk) pair fits "
                f"within the token limit {ce_max_tokens}."
            )
            return None

        batch = [(gen_sentences[gi], orig_chunks[ci]) for gi, ci in valid_pairs]
        with self._lock:
            scores = self.model.predict(batch)

        gen_max_sims: List[float] = [0.0] * len(gen_sentences)
        for (gi, _), score in zip(valid_pairs, scores):
            if gi < len(gen_max_sims):
                gen_max_sims[gi] = max(gen_max_sims[gi], float(score))

        matched_count = sum(1 for s in gen_max_sims if s > 0.0)
        if matched_count > 0:
            return float(sum(gen_max_sims) / len(gen_max_sims))
        return None

    def calculate_similarity(
        self,
        gen_text: Optional[str],
        orig_text: Optional[str],
        ce_model_max_tokens: int = 512,
        special_tokens: int = 2,
    ) -> Optional[float]:
        """
        Calculate similarity score between generated text and original text.
        """
        if (
            not gen_text
            or not orig_text
            or not gen_text.strip()
            or not orig_text.strip()
        ):
            return None

        gen_text = gen_text.strip()
        orig_text = orig_text.strip()

        tokenizer = self.tokenizer
        tokens_s1 = len(tokenizer.encode(gen_text))
        tokens_s2 = len(tokenizer.encode(orig_text))

        if tokens_s1 + tokens_s2 + special_tokens > ce_model_max_tokens:
            return self._calculate_similarity_patch_match(
                gen_text=gen_text,
                orig_text=orig_text,
                ce_max_tokens=ce_model_max_tokens,
                special_tokens=special_tokens,
            )

        with self._lock:
            scores = self.model.predict([(gen_text, orig_text)])

        if len(scores) > 0:
            return float(scores[0])
        return 0.0


def create_app(service: Optional[CrossEncoderService] = None) -> Flask:
    """
    Factory function for Flask app.
    """
    app = Flask(__name__)
    ce_service = service or CrossEncoderService()

    @app.route("/health", methods=["GET"])
    @app.route("/status", methods=["GET"])
    def health_check():
        return (
            jsonify(
                {
                    "status": "ok",
                    "model_name": ce_service.model_name,
                }
            ),
            200,
        )

    @app.route("/calculate_similarity", methods=["POST"])
    @app.route("/api/calculate_similarity", methods=["POST"])
    @app.route("/similarity", methods=["POST"])
    def calculate_similarity_endpoint():
        data = request.get_json(silent=True) or {}

        gen_text = data.get("gen_text") or data.get("gen_article_str")
        orig_text = data.get("orig_text") or data.get("orig_article_str")
        ce_model_max_tokens = int(data.get("ce_model_max_tokens", 512))
        special_tokens = int(data.get("special_tokens", 2))

        if gen_text is None or orig_text is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Missing 'gen_text' or 'orig_text' in request body.",
                        "similarity": None,
                    }
                ),
                400,
            )

        try:
            sim_val = ce_service.calculate_similarity(
                gen_text=gen_text,
                orig_text=orig_text,
                ce_model_max_tokens=ce_model_max_tokens,
                special_tokens=special_tokens,
            )
            return (
                jsonify(
                    {
                        "status": "ok",
                        "similarity": sim_val,
                    }
                ),
                200,
            )
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}", exc_info=True)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": str(e),
                        "similarity": None,
                    }
                ),
                500,
            )

    return app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    port = int(os.environ.get("CROSS_ENCODER_PORT", 8085))
    host = os.environ.get("CROSS_ENCODER_HOST", "0.0.0.0")

    app = create_app()
    logger.info(f"Starting Cross-Encoder API on {host}:{port}...")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
