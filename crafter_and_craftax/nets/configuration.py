from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT2WorldModelConfig(GPT2Config):
    def __init__(
        self,
        num_actions=17,
        tokens_per_block=81,
        max_blocks=21,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_actions = num_actions
        self.tokens_per_block = tokens_per_block
        self.max_blocks = max_blocks

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

    def __hash__(self):
        return hash(self.to_json_string())
