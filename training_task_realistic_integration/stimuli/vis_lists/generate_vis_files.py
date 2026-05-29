import pandas as pd
import shutil
import os

# Configuration
num_participants = 60  # Adjust as needed
start_participant = 101  # Starting participant number
scenes = ['JA', 'JB', 'UA', 'UB']
labels = {'JA': 'Tulon', 'JB': 'Malbow', 'UA': 'Negvi', 'UB': 'Somar'}

# TWO images per scene (side-by-side)
images_left = {
    'JA': 'TESTstimuli/TESTscenes/JA_1.jpg', 
    'JB': 'TESTstimuli/TESTscenes/JB_1.jpg',
    'UA': 'TESTstimuli/TESTscenes/UA_1.jpg',
    'UB': 'TESTstimuli/TESTscenes/UB_1.jpg'
}
images_right = {
    'JA': 'TESTstimuli/TESTscenes/JA_3.jpg', 
    'JB': 'TESTstimuli/TESTscenes/JB_3.jpg',
    'UA': 'TESTstimuli/TESTscenes/UA_3.jpg',
    'UB': 'TESTstimuli/TESTscenes/UB_3.jpg'
}

# Create vis_lists folder if it doesn't exist
os.makedirs('vis_lists', exist_ok=True)

# Generate files for each participant
for participant in range(start_participant, start_participant + num_participants):
    for scene in scenes:
        # Create filename
        filename = f'vis_lists/visualization_{scene}_sub_{participant}.csv'
        
        # Copy template
        shutil.copyfile('visualization_template.csv', filename)
        
        # Add scene-specific columns (NOW WITH TWO IMAGES)
        df = pd.read_csv(filename)
        df['label_current'] = labels[scene]
        df['scene_left'] = images_left[scene]
        df['scene_right'] = images_right[scene]
        
        # Save
        df.to_csv(filename, index=False)
        
    print(f'Created files for participant {participant}')

print(f'\nDone! Created {num_participants * 4} files for {num_participants} participants')