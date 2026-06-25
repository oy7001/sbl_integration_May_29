import pandas as pd
import shutil
import os

# Ensure script runs relative to its own folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Participant range
start_participant = 233
end_participant = 297

# Ensure output folder exists
os.makedirs('vis_lists', exist_ok=True)

for participant in range(start_participant, end_participant + 1):

    # Load participant condition list
    cond_filename = f'TESTstimuli/cond_lists/scene_names_sub_{participant}.csv'
    df_cond = pd.read_csv(cond_filename)

    # Loop through each scene for this participant
    for scene in df_cond['scene_stimulus'].unique():

        # Get the correct label from condition list
        label = df_cond.loc[df_cond['scene_stimulus'] == scene, 'scene_label'].iloc[0]

        # Build image paths
        scene_left = f'TESTstimuli/TESTscenes/{scene}_1.jpg'
        scene_right = f'TESTstimuli/TESTscenes/{scene}_2.jpg'

        # Template file (contains the questions)
        template_path = f'TESTstimuli/vis_templates/visualization_{scene}_template.csv'

        # Output visualization file
        vis_filename = f'TESTstimuli/vis_lists/visualization_{scene}_sub_{participant}.csv'

        # Copy template
        shutil.copyfile(template_path, vis_filename)

        # Load copied file
        df_vis = pd.read_csv(vis_filename)

        # Fill columns
        df_vis['label_current'] = label
        df_vis['scene_left'] = scene_left
        df_vis['scene_right'] = scene_right

        # Save updated file
        df_vis.to_csv(vis_filename, index=False)

    print(f'Created visualization files for participant {participant}')

print("\nDone! Visualization lists generated.")