from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_PATH = "backend/models/resume_matcher"

def load_data(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                InputExample(
                    texts=[row["resume"], row["job"]],
                    label=float(row["label"])
                )
            )
    return examples

def train():
    model = SentenceTransformer(MODEL_NAME)
    data = load_data("backend/data/training_data.jsonl")

    dataloader = DataLoader(data, shuffle=True, batch_size=8)
    loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=3,
        warmup_steps=10
    )

    model.save(OUTPUT_PATH)
    print("Model trained and saved")

if __name__ == "__main__":
    train()
