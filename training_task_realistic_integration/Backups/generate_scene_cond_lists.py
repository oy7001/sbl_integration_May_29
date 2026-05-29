import pandas as pd
import random
import os

# where participant files will be saved
output_folder = "TESTstimuli/cond_lists"
os.makedirs(output_folder, exist_ok=True)

# scene IDs (your naming)
scene_ids = ['JA','JB','UA','UB']

# scene label pool
labels = ['Negvi','Somar','Malbow','Tulon']

# correct keys (match your experiment)
correct_keys = {
    'JA':1,
    'JB':2,
    'UA':3,
    'UB':4
}

participants = range(1,101)

for p in participants:

    # randomize scene labels
    shuffled_labels = random.sample(labels, len(labels))

    rows = []

    for i, scene in enumerate(scene_ids):

        # randomly choose one of the three images
        img_number = random.choice([1,2,3])

        scene_img = f"TESTstimuli/TESTscenes/{scene}_{img_number}.jpg"

        rows.append({
            'scene_stimulus': scene,
            'scene_label': shuffled_labels[i],
            'scene_img': scene_img,
            'correct_key': correct_keys[scene]
        })

    df = pd.DataFrame(rows)

    filename = f"{output_folder}/scene_names_sub_{p}.csv"
    df.to_csv(filename,index=False)

    print(f"Created {filename}")