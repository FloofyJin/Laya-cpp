import json
import os

import torch

from laya.common import build_model


class GoldenModel:
    def __init__(self, model_dir):
        from safetensors.torch import load_file

        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        enc_dir = os.path.join(model_dir, "encoder")
        self.model = build_model(self.cfg, encoder_dir=enc_dir if os.path.isdir(enc_dir) else None,
                                 pretrained=False)
        weights = load_file(os.path.join(model_dir, "model.safetensors"))
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()

    def embeddings(self, input_ids):
        ids = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.encoder.embeddings(input_ids=ids)
        return out[0].numpy()

    def hidden_after_layer(self, input_ids, layer_idx):
        ids = torch.tensor([input_ids], dtype=torch.long)
        captured = {}

        def hook(module, inputs, output):
            captured["out"] = output

        handle = self.model.encoder.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model.encoder(input_ids=ids)
        finally:
            handle.remove()
        return captured["out"][0].numpy()

    def encoder_output(self, input_ids):
        ids = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.encoder(input_ids=ids)
        return out.last_hidden_state[0].numpy()
