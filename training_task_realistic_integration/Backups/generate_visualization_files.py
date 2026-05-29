import pandas as pd
import os

# folders
cond_folder = "TESTstimuli/cond_lists"
template_folder = "TESTstimuli/vis_templates"
output_folder = "TESTstimuli/vis_lists"

participants = range(1,101)

for p in participants:

    scene_file = f"{cond_folder}/scene_names_sub_{p}.csv"
    df_scene = pd.read_csv(scene_file)

    # create dictionaries for label and image lookup
    label_dict = dict(zip(df_scene['scene_stimulus'], df_scene['scene_label']))
    img_dict = dict(zip(df_scene['scene_stimulus'], df_scene['scene_img']))

    # make participant folder
    sub_folder = f"{output_folder}/sub_{p}"
    os.makedirs(sub_folder, exist_ok=True)

    for scene in ['JA','JB','UA','UB']:

        template_file = f"{template_folder}/visualization_{scene}_template.csv"
        df_vis = pd.read_csv(template_file)

        # insert correct label and image
        df_vis['label_current'] = label_dict.get(scene,"")
        df_vis['scene_again'] = img_dict.get(scene,"")

        out_file = f"{sub_folder}/visualization_{scene}_sub_{p}.csv"
        df_vis.to_csv(out_file,index=False)

        print(f"Created {out_file}")