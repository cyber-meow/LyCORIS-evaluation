import json
import os

import torch
import torch.nn as nn
from transformers import CLIPModel


class AestheticScorer(nn.Module):

    def __init__(self,
                 clip_model_name,
                 input_size=1024,
                 use_activation=False,
                 dropout=0,
                 hidden_dim=1024,
                 reduce_dims=False,
                 output_activation=None):
        super().__init__()

        self.config = {
            "input_size": input_size,
            "use_activation": use_activation,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "reduce_dims": reduce_dims,
            "output_activation": output_activation
        }

        self.clip_vision_model = (
            CLIPModel.from_pretrained(clip_model_name).vision_model)

        layers = [
            nn.Linear(self.config["input_size"], self.config["hidden_dim"]),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                self.config["hidden_dim"],
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1))),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (2 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1))),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Dropout(self.config["dropout"]),
            nn.Linear(
                round(self.config["hidden_dim"] / (4 if reduce_dims else 1)),
                round(self.config["hidden_dim"] / (8 if reduce_dims else 1))),
            nn.ReLU() if self.config["use_activation"] else None,
            nn.Linear(
                round(self.config["hidden_dim"] / (8 if reduce_dims else 1)),
                1),
        ]
        if self.config["output_activation"] == "sigmoid":
            layers.append(nn.Sigmoid())
        layers = [x for x in layers if x is not None]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            vision_output = self.clip_vision_model(x)
        embeddings = vision_output.pooler_output
        x = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        if self.config["output_activation"] == "sigmoid":
            upper, lower = 10, 1
            scale = upper - lower
            return (self.layers(x) * scale) + lower
        else:
            return self.layers(x)

    def save(self, save_name):
        split_name = os.path.splitext(save_name)
        with open(f"{split_name[0]}.config", "w") as outfile:
            outfile.write(json.dumps(self.config, indent=4))

        for i in range(
                6
        ):  # saving sometiles fails, so retry 5 times, might be windows issue
            try:
                torch.save(self.state_dict(), save_name)
                break
            except RuntimeError as e:
                # check if error contains string "File"
                if "cannot be opened" in str(e) and i < 5:
                    print("Model save failed, retrying...")
                else:
                    raise e
