import sys
import os

# Add the directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_results_mrc_knn import mrc2prompt

# Simulate the scenario where example_idx might be shorter than mrc_data
mrc_data = []
for i in range(50):
    mrc_data.append({
        "context": f"Test sentence {i}",
        "entity_label": "PER",
        "start_position": [],
        "end_position": []
    })

train_mrc_data = [
    {
        "context": "Training example 1",
        "start_position": [],
        "end_position": []
    }
]

# example_idx has fewer entries than mrc_data
example_idx = [[0] for _ in range(30)]  # Only 30 entries but mrc_data has 50

print(f"mrc_data length: {len(mrc_data)}")
print(f"example_idx length: {len(example_idx)}")
print(f"train_mrc_data length: {len(train_mrc_data)}")

try:
    prompts = mrc2prompt(
        mrc_data=mrc_data,
        data_name="CONLL",
        example_idx=example_idx,
        train_mrc_data=train_mrc_data,
        example_num=1,
        last_results=None
    )
    print(f"Success! Generated {len(prompts)} prompts")
except IndexError as e:
    print(f"IndexError occurred: {e}")
    import traceback
    traceback.print_exc()
