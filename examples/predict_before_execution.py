"""A new task's decision requires its embedding, not its execution outcomes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skilldelta import predict_gain

prediction = predict_gain(
    query_vectors=[[0.95, 0.05]], query_ids=["new_task"], query_families=["calculator"],
    support_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
    support_ids=["history_1", "history_2", "history_3"],
    support_families=["calculator", "calculator", "calculator"],
    skip_outcomes=[0.0, 1.0, 1.0], use_outcomes=[1.0, 1.0, 0.0],
)
print("Predicted gain:", round(float(prediction["scores"][0]), 4))
print("Action:", "Use skill" if prediction["use_skill"][0] else "Skip skill")
