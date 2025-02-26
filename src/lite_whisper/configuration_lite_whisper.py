from transformers import WhisperConfig

class LiteWhisperConfig(WhisperConfig):
    model_type = "lite-whisper"

    def __init__(
        self, 
        low_rank_config: list[dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.low_rank_config = low_rank_config
