import os
import pandas as pd
import random

STIM_DIR = "TESTstimuli/stim_lists"
template = os.path.join(STIM_DIR,"integrated_stim_template.csv")

SUBS = range(233,297)

scene_names = ["Negvi","Tulon","Malbow","Somar"]

for sub in SUBS:

    names = scene_names.copy()
    random.shuffle(names)

    label_map = {
        "JA":names[0],
        "JB":names[1],
        "UA":names[2],
        "UB":names[3]
    }

    df = pd.read_csv(template)

    scene_stim = []
    scene_labels = []

    jungle_counter = 0
    sea_counter = 0

    for cat in df["scene_cat"]:

        if cat == "Jungle":
            scene = ["JA","JB"][jungle_counter % 2]
            jungle_counter += 1
        else:
            scene = ["UA","UB"][sea_counter % 2]
            sea_counter += 1

        scene_stim.append(f"stimuli/scenes/scene_{scene}.png")
        scene_labels.append(label_map[scene])

    df["scene_stimulus"] = scene_stim
    df["scene_label"] = scene_labels

    out = os.path.join(STIM_DIR,f"stimuli_list_training_sub_{sub}.csv")

    df.to_csv(out,index=False)

print("Training stim lists generated.")