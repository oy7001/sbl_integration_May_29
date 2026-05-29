# Olivia_generate_all_lists.py
#
# Unified script that generates all participant lists for the experiment.
# Run order:
#   1. create_stimuli_lists        – generates per-participant encoding / retrieval stim lists
#   2. create_training_cond_lists  – extracts scene-name mappings into training cond lists
#   3. create_training_vis_and_stim_lists – copies templates and fills in scene labels
#
# Template files required before running step 3:
#   - <TRAINING_VIS_LISTS_DIR>/visualization_<scene>_template.csv  (for scene in JA,JB,UA,UB)
#   - <TRAINING_STIM_LISTS_DIR>/stimuli_list_training_template.csv

import os
import shutil
import random
from itertools import combinations

import pandas as pd
import numpy as np

# ============================================================
# PATH CONFIGURATION  – edit these for each new experiment variant
# ============================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
print("BASE_PATH:", BASE_PATH)


# Experiment folder names
TRAINING_TASK_DIR  = os.path.join(BASE_PATH, 'training_task_realistic_integration') # same for all behav/MEG variants
ENCODING_EXP_DIR   = os.path.join(BASE_PATH, 'encoding_task_behav') # change according to experiment variant (e.g. encoding_task_behav; encoding_task_meg)
RETRIEVAL_EXP_DIR  = os.path.join(BASE_PATH, 'retrieval_task_behav') # change according to experiment variant (e.g. retrieval_task_behav; retrieval_task_meg)

# Set to True to include MEG trigger values in the encoding stim lists
IS_MEG = False # MAKE SURE THIS MATCHES THE EXPERIMENT VARIANT YOU'RE GENERATING LISTS FOR (e.g. True for encoding_task_meg; False for encoding_task_behav)

# Sub-paths derived from the experiment folders (no need to change normally)
ENCODING_STIM_LISTS_DIR   = os.path.join(ENCODING_EXP_DIR,  'stimuli', 'stim_lists')
RETRIEVAL_STIM_LISTS_DIR  = os.path.join(RETRIEVAL_EXP_DIR, 'stimuli', 'stim_lists')
TRAINING_COND_LISTS_DIR   = os.path.join(TRAINING_TASK_DIR,  'stimuli', 'cond_lists')
TRAINING_VIS_LISTS_DIR    = os.path.join(TRAINING_TASK_DIR,  'stimuli', 'vis_lists')
TRAINING_STIM_LISTS_DIR   = os.path.join(TRAINING_TASK_DIR,  'stimuli', 'stim_lists')

# Norming data
NORMING_DIR       = os.path.join(BASE_PATH, 'analyses', 'norming_exp')
ITEM_NORMING_FILE = os.path.join(NORMING_DIR, 'dict_norming_to_encoding_rating_data_final_80_set.csv')
MEMORABILITY_FILE = os.path.join(NORMING_DIR, 'item_memorability_onesession.csv')


# Subject range (3 digits for behavioral participants, 4 digits for MEG participants)
SUBS = range(300, 365)  # Inclusive range of subject numbers (e.g. 301 to 333 for 32 subjects) - each 32 subjects are counterbalanced based on counterbalancing table.

# Suffix used to identify participant-neutral template files
TEMPLATE_SUFFIX = 'template'


# ============================================================
# SHARED HELPERS
# ============================================================

def pseudo_random_order(df, column):
    """Shuffle df so that no two consecutive rows share the same value in `column`."""
    shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    def swap_rows(df, i, j):
        df.iloc[[i, j]] = df.iloc[[j, i]].values

    for i in range(len(shuffled_df) - 1):
        if shuffled_df.at[i, column] == shuffled_df.at[i + 1, column]:
            for j in range(i + 2, len(shuffled_df)):
                if shuffled_df.at[j, column] != shuffled_df.at[i, column]:
                    swap_rows(shuffled_df, i + 1, j)
                    break

    for i in range(len(shuffled_df) - 1):
        if shuffled_df.at[i, column] == shuffled_df.at[i + 1, column]:
            return pseudo_random_order(df, column)

    return shuffled_df


def split_balanced_groups(items_df):
    """
    Split items_df into two groups with similar means for t_stat and prediction.
    Returns (group1_df, group2_df).
    """
    if 't_stat' not in items_df.columns or 'prediction' not in items_df.columns:
        raise ValueError("DataFrame must contain 't_stat' and 'prediction' columns")

    n_items = len(items_df)
    half_size = n_items // 2

    if n_items % 2 != 0:
        print(f"Warning: Odd number of items ({n_items}). One group will have one more item.")

    t_stat_std = items_df['t_stat'].std() or 1
    pred_std   = items_df['prediction'].std() or 1

    def calc_difference(indices1, indices2):
        g1, g2 = items_df.iloc[list(indices1)], items_df.iloc[list(indices2)]
        t_diff = abs((g1['t_stat'].mean() - g2['t_stat'].mean()) / t_stat_std)
        p_diff = abs((g1['prediction'].mean() - g2['prediction'].mean()) / pred_std)
        return t_diff + p_diff

    indices   = list(range(n_items))
    best_diff = float('inf')
    best_split = None

    for combo in combinations(indices, half_size):
        remaining = set(indices) - set(combo)
        diff = calc_difference(combo, remaining)
        if diff < best_diff:
            best_diff  = diff
            best_split = (list(combo), list(remaining))

    group1_df = items_df.iloc[best_split[0]].copy()
    group2_df = items_df.iloc[best_split[1]].copy()

    print(f"  Group 1: {len(group1_df)} items, "
          f"t_stat mean = {group1_df['t_stat'].mean():.4f}, "
          f"prediction mean = {group1_df['prediction'].mean():.4f}")
    print(f"  Group 2: {len(group2_df)} items, "
          f"t_stat mean = {group2_df['t_stat'].mean():.4f}, "
          f"prediction mean = {group2_df['prediction'].mean():.4f}")

    return group1_df, group2_df

def normalize_filename(path):
    return os.path.basename(str(path)).strip().lower()
# ============================================================
# STEP 1 – Create per-participant encoding & retrieval stim lists
# ============================================================

def create_stimuli_lists(subs, encoding_stim_lists_dir, retrieval_stim_lists_dir,
                         item_norming_file, memorability_file=None, is_meg=True):
    """
    For each subject in `subs`:
      - Randomises scene-name assignments and SC/SI item splits
      - Creates 3 pseudo-randomised runs and saves to encoding_stim_lists_dir
      - Copies run_1 to retrieval_stim_lists_dir
    """
    print("\n=== STEP 1: Creating encoding & retrieval stimuli lists ===")

    

    # --- Load & prepare norming data ---

    item_norming = pd.read_csv(item_norming_file)

    item_norming = item_norming[
    ['item_cat',
     'item_type',
     'encoding_item_stimulus',
     't_stat']
]

    print(
        item_norming.groupby(
            ['item_cat', 'item_type']
        ).size()
    )

    # --- Memorability merge 
    memorability_scores = pd.read_csv(memorability_file)
    memorability_scores.columns = memorability_scores.columns.str.strip()

    # rename
    memorability_scores = memorability_scores.rename(
        columns={'filename': 'encoding_item_stimulus'}
    )

    # --- Normalize filenames for safe merge ---
    item_norming['merge_key'] = item_norming['encoding_item_stimulus'].apply(normalize_filename)
    memorability_scores['merge_key'] = memorability_scores['encoding_item_stimulus'].apply(normalize_filename)

    # add path ONCE
    memorability_scores['encoding_item_stimulus'] = (
        'stimuli/items/' + memorability_scores['encoding_item_stimulus']
    )
    
    item_norming = pd.merge(
        item_norming,
        memorability_scores[['merge_key', 'prediction']],
        on='merge_key',
        how='left'
    )
    print("Missing predictions:",
      item_norming['prediction'].isna().sum())

    print(item_norming[
        ['encoding_item_stimulus', 'prediction']
    ].head())

    missing = item_norming[item_norming['prediction'].isna()]

    print("\n=== MISSING MATCHES ===")
    print(missing['encoding_item_stimulus'].head(20))

    print("\n=== MEMORABILITY SAMPLE ===")
    print(memorability_scores['encoding_item_stimulus'].head(20))

    print("NaN rate:", item_norming['prediction'].isna().mean())

    nan_rate = item_norming['prediction'].isna().mean()
    print("NaN rate:", nan_rate)

    if nan_rate > 0.6:
        raise ValueError("Merge truly failed — most items missing")

    # Fill missing values with mean (SAFE + standard)
    item_norming['prediction'] = item_norming['prediction'].fillna(
        item_norming['prediction'].mean()
    )

    # --- Scene definitions ---
    scenes = [
        'stimuli/scenes/JA_1.jpg',
        'stimuli/scenes/JB_1.jpg',
        'stimuli/scenes/UA_1.jpg',
        'stimuli/scenes/UB_1.jpg'
    ]

    scene_ids = ['JA', 'JB', 'UA', 'UB']
    scene_cat = ['Jungle', 'Jungle', 'Undersea', 'Undersea']

    # --- Pre-compute balanced item splits (same across all subjects) ---
    print("Computing balanced item splits...")
    jungle_objects = item_norming[(item_norming['item_cat'] == 'Jungle') &
                                   (item_norming['item_type'] == 'object')].reset_index(drop=True)
    jungle_objects_1, jungle_objects_2 = split_balanced_groups(jungle_objects)

    jungle_animals = item_norming[(item_norming['item_cat'] == 'Jungle') &
                                   (item_norming['item_type'] == 'animal')].reset_index(drop=True)
    jungle_animals_1, jungle_animals_2 = split_balanced_groups(jungle_animals)

    sea_objects = item_norming[(item_norming['item_cat'] == 'Undersea') &
                                (item_norming['item_type'] == 'object')].reset_index(drop=True)
    sea_objects_1, sea_objects_2 = split_balanced_groups(sea_objects)

    sea_animals = item_norming[(item_norming['item_cat'] == 'Undersea') &
                                (item_norming['item_type'] == 'animal')].reset_index(drop=True)
    sea_animals_1, sea_animals_2 = split_balanced_groups(sea_animals)

    # --- MEG trigger dictionary ---
    trig_dict = {
        'J_j_ob': 1, 'J_j_an': 2, 'J_u_ob': 3, 'J_u_an': 4,
        'U_u_ob': 5, 'U_u_an': 6, 'U_j_ob': 7, 'U_j_an': 8,
    }

    # --- Per-subject loop ---
    for sub in subs:

        scene_names = ['Negvi', 'Tulon', 'Malbow', 'Somar']
        random.shuffle(scene_names)

        df = pd.DataFrame(columns=[
            'scene_stimulus',
            'scene_cat',
            'scene_label',
            'canonical_item',
            'item_cat',
            'item_type',
            'congruency'
        ])
        df['scene_stimulus'] = scenes * 20
        df['scene_label']    = scene_names * 20
        df['scene_cat']      = scene_cat * 20
        df = df.sort_values(by=['scene_cat', 'scene_stimulus']).reset_index(drop=True)

        # Random SC/SI assignment for each item category
        np.random.seed(1000 + sub)
        rand_vec = np.random.randint(0, 2, size=4)

        def pick(group1, group2, flag):
            return (group1, group2) if flag == 0 else (group2, group1)

        sc_jo, si_jo = pick(jungle_objects_1, jungle_objects_2, rand_vec[0])
        sc_ja, si_ja = pick(jungle_animals_1, jungle_animals_2, rand_vec[1])
        sc_so, si_so = pick(sea_objects_1,    sea_objects_2,    rand_vec[2])
        sc_sa, si_sa = pick(sea_animals_1,    sea_animals_2,    rand_vec[3])

        sc_jo = sc_jo['encoding_item_stimulus'].tolist()
        sc_ja = sc_ja['encoding_item_stimulus'].tolist()
        sc_so = sc_so['encoding_item_stimulus'].tolist()
        sc_sa = sc_sa['encoding_item_stimulus'].tolist()
        si_jo = si_jo['encoding_item_stimulus'].tolist()
        si_ja = si_ja['encoding_item_stimulus'].tolist()
        si_so = si_so['encoding_item_stimulus'].tolist()
        si_sa = si_sa['encoding_item_stimulus'].tolist()

        for lst in [sc_jo, sc_ja, sc_so, sc_sa, si_jo, si_ja, si_so, si_sa]:
            random.shuffle(lst)

       # Allocate to scenes
        sc_jungle_scene1 = sc_jo[:5] + sc_ja[:5]
        sc_jungle_scene2 = sc_jo[5:10] + sc_ja[5:10]

        sc_sea_scene1    = sc_so[:5] + sc_sa[:5]
        sc_sea_scene2    = sc_so[5:10] + sc_sa[5:10]

        si_jungle_scene1 = si_so[:5] + si_sa[:5]
        si_jungle_scene2 = si_so[5:10] + si_sa[5:10]

        si_sea_scene1    = si_jo[:5] + si_ja[:5]
        si_sea_scene2    = si_jo[5:10] + si_ja[5:10]
        print(len(sc_jo), len(sc_ja))
        # canonical_item stores the finalized encoding filename selected
        # from dict_norming_to_encoding_rating_data_final_80_set.csv

        df.loc[0:9,   'canonical_item'] = sc_jungle_scene1;  df.loc[0:9,   'congruency'] = 'sc'
        df.loc[10:19, 'canonical_item'] = si_jungle_scene1;  df.loc[10:19, 'congruency'] = 'si'
        df.loc[20:29, 'canonical_item'] = sc_jungle_scene2;  df.loc[20:29, 'congruency'] = 'sc'
        df.loc[30:39, 'canonical_item'] = si_jungle_scene2;  df.loc[30:39, 'congruency'] = 'si'
        df.loc[40:49, 'canonical_item'] = sc_sea_scene1;     df.loc[40:49, 'congruency'] = 'sc'
        df.loc[50:59, 'canonical_item'] = si_sea_scene1;     df.loc[50:59, 'congruency'] = 'si'
        df.loc[60:69, 'canonical_item'] = sc_sea_scene2;     df.loc[60:69, 'congruency'] = 'sc'
        df.loc[70:79, 'canonical_item'] = si_sea_scene2;     df.loc[70:79, 'congruency'] = 'si'

        df.loc[df['canonical_item'].str.contains('jungle'), 'item_cat'] = 'Jungle'
        df.loc[df['canonical_item'].str.contains('undersea'), 'item_cat'] = 'Undersea'
        df['item_type'] = 'object'
        df.loc[df['canonical_item'].str.contains('animal'), 'item_type'] = 'animal'

       # --- Build integrated stimulus paths ---

        scene_prefix_map = {
            'JA': 'jungle_a',
            'JB': 'jungle_b',
            'UA': 'ocean_a',
            'UB': 'ocean_b'
        }

        def build_integrated_path(scene_stimulus, canonical_item):

            scene_id = os.path.basename(scene_stimulus).split('_')[0]
            scene_prefix = scene_prefix_map[scene_id]

            filename = os.path.basename(canonical_item)
            filename = filename.replace('.png', '.jpg').replace('.jpeg', '.jpg')

            return (
                f"stimuli/integrated/"
                f"{scene_prefix}_{filename}"
            )

        integrated_list = []

        for _, row in df.iterrows():

            integrated_list.append(
                build_integrated_path(
                    row['scene_stimulus'],
                    row['canonical_item']
                )
            )

        df['integrated_stimulus'] = integrated_list


        # --- Build recalled stimulus paths ---

        def build_recalled_path(integrated_stimulus):

            filename = os.path.basename(integrated_stimulus)

            return (
                f"stimuli/recalled/"
                f"recalled_{filename}"
            )

        df['recalled_stimulus'] = df['integrated_stimulus'].apply(
            build_recalled_path
        )

        if is_meg:
            df['trigger_value'] = np.nan
            j, u = 'Jungle', 'Undersea'

            df.loc[(df['scene_cat']==j)&(df['item_cat']==j)&(df['item_type']=='object'), 'trigger_value'] = trig_dict['J_j_ob']
            df.loc[(df['scene_cat']==j)&(df['item_cat']==j)&(df['item_type']=='animal'), 'trigger_value'] = trig_dict['J_j_an']
            df.loc[(df['scene_cat']==j)&(df['item_cat']==u)&(df['item_type']=='object'), 'trigger_value'] = trig_dict['J_u_ob']
            df.loc[(df['scene_cat']==j)&(df['item_cat']==u)&(df['item_type']=='animal'), 'trigger_value'] = trig_dict['J_u_an']

            df.loc[(df['scene_cat']==u)&(df['item_cat']==u)&(df['item_type']=='object'), 'trigger_value'] = trig_dict['U_u_ob']
            df.loc[(df['scene_cat']==u)&(df['item_cat']==u)&(df['item_type']=='animal'), 'trigger_value'] = trig_dict['U_u_an']
            df.loc[(df['scene_cat']==u)&(df['item_cat']==j)&(df['item_type']=='object'), 'trigger_value'] = trig_dict['U_j_ob']
            df.loc[(df['scene_cat']==u)&(df['item_cat']==j)&(df['item_type']=='animal'), 'trigger_value'] = trig_dict['U_j_an']

            df['trigger_value'] = df['trigger_value'].astype(int)

        for run in range(1, 4):
            ordered_df = pseudo_random_order(df, 'scene_label')
            out_file = os.path.join(
                encoding_stim_lists_dir,
                f'stimuli_list_sub_{sub}_run_{run}.csv'
            )
            ordered_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file}")

       # Copy all runs to retrieval folder

        for run in range(1, 4):

            src = os.path.join(
                encoding_stim_lists_dir,
                f'stimuli_list_sub_{sub}_run_{run}.csv'
            )

            dst = os.path.join(
                retrieval_stim_lists_dir,
                f'stimuli_list_sub_{sub}_run_{run}.csv'
            )

            shutil.copyfile(src, dst)

            print(f"  Copied run_{run} to retrieval: {dst}")

    print("Step 1 complete.")


# ============================================================
# STEP 2 – Create training condition (scene-name) lists
# ============================================================

def _extract_scene_label_pairs(df):
    """Return dict mapping scene identifier (JA/JB/UA/UB) → scene_label."""
    pairs = {}
    for _, row in df.iterrows():
        if pd.notna(row['scene_label']):
            identifier = os.path.basename(row['scene_stimulus']).split('_')[0]
            pairs[identifier] = row['scene_label']
    return pairs


def create_training_cond_lists(subs, encoding_stim_lists_dir, training_cond_lists_dir):
    """
    For each subject in `subs`:
      Reads the encoding stim list (run_1) to extract scene-name assignments,
      then writes scene_names_sub_{sub}.csv to training_cond_lists_dir.
    """
    print("\n=== STEP 2: Creating training condition lists ===")

    scene_identifiers = ['JA', 'JB', 'UA', 'UB']
    scene_imgs = [f'stimuli/scenes/{s}_1.jpg' for s in scene_identifiers]
    correct_keys      = ['1', '2', '3', '4']

    for sub in subs:
        enc_file = os.path.join(encoding_stim_lists_dir, f'stimuli_list_sub_{sub}_run_1.csv')
        if not os.path.exists(enc_file):
            print(f"  WARNING: encoding stim list not found for sub {sub}: {enc_file}")
            continue

        df_enc = pd.read_csv(enc_file)
        pairs  = _extract_scene_label_pairs(df_enc)

        scene_labels = [pairs.get(sid) for sid in scene_identifiers]

        new_df = pd.DataFrame({
            'scene_stimulus': scene_identifiers,
            'scene_label':    scene_labels,
            'scene_img':      scene_imgs,
            'correct_key':    correct_keys,
        })

        out_file = os.path.join(training_cond_lists_dir, f'scene_names_sub_{sub}.csv')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        new_df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file}")

    print("Step 2 complete.")


# ============================================================
# STEP 3 – Create training visualisation and stimuli lists
# ============================================================

def create_training_vis_and_stim_lists(subs, training_cond_lists_dir,
                                       training_vis_lists_dir, training_stim_lists_dir,
                                       template_suffix=TEMPLATE_SUFFIX):
    """
    For each subject in `subs`:
      1. Copies visualization_<scene>_template.csv → visualization_<scene>_sub_{sub}.csv
         and fills in label_current and scene_again columns from the cond list.
      2. Copies stimuli_list_training_template.csv → stimuli_list_training_sub_{sub}.csv
         and fills in the scene_label column from the cond list.
    """
    print("\n=== STEP 3: Creating training vis and stim lists ===")

    for sub in subs:
        cond_file = os.path.join(training_cond_lists_dir, f'scene_names_sub_{sub}.csv')
        if not os.path.exists(cond_file):
            print(f"  WARNING: cond list not found for sub {sub}: {cond_file}")
            continue

        df_cond = pd.read_csv(cond_file)
        label_dict = dict(zip(df_cond['scene_stimulus'], df_cond['scene_label']))
        img_dict   = dict(zip(df_cond['scene_stimulus'], df_cond['scene_img']))

        # --- Visualisation files ---
        for scene in ['JA', 'JB', 'UA', 'UB']:
            template_vis = os.path.join(TRAINING_TASK_DIR,
                           'stimuli',
                           'vis_templates',
                           f'visualization_{scene}_{template_suffix}.csv')
            if not os.path.exists(template_vis):
                print(f"  WARNING: vis template not found: {template_vis}")
                continue
            os.makedirs(training_vis_lists_dir, exist_ok=True)
            dst_vis = os.path.join(training_vis_lists_dir,
                                   f'visualization_{scene}_sub_{sub}.csv')
            shutil.copyfile(template_vis, dst_vis)
            df_vis = pd.read_csv(dst_vis)

            #FIX: remove bad columns
            df_vis = df_vis.loc[:, ~df_vis.columns.str.contains('^Unnamed')]
            df_vis = df_vis.loc[:, df_vis.columns.notna()]

            df_vis['label_current'] = label_dict.get(scene, "")
            df_vis['scene_left']  = f'stimuli/scenes/{scene}_1.jpg'
            df_vis['scene_right'] = f'stimuli/scenes/{scene}_2.jpg'

            # --- FORCE INTEGER TYPES (STRICT, NO SILENT FIXES) ---
            if 'qst_num' in df_vis.columns:
                df_vis['qst_num'] = pd.to_numeric(df_vis['qst_num'], errors='raise').astype(int)

            if 'correct_answer' in df_vis.columns:
                df_vis['correct_answer'] = pd.to_numeric(df_vis['correct_answer'], errors='raise').astype(int)

            df_vis.to_csv(dst_vis, index=False)
            print(f"  Updated vis file: {dst_vis}")

        # --- Training stim list ---
        template_stim = os.path.join(training_stim_lists_dir,
                                     'integrated_stim_template.csv')
        if not os.path.exists(template_stim):
            print(f"  WARNING: stim template not found for sub {sub}: {template_stim}")
            continue
        os.makedirs(training_stim_lists_dir, exist_ok=True)
        dst_stim = os.path.join(training_stim_lists_dir,
                                f'stimuli_list_training_sub_{sub}.csv')
        shutil.copyfile(template_stim, dst_stim)

        df_stim = pd.read_csv(dst_stim)

        def extract_scene_from_integrated(path):

            filename = os.path.basename(path)

            if filename.startswith('jungle_a'):
                return 'JA'

            elif filename.startswith('jungle_b'):
                return 'JB'

            elif filename.startswith('ocean_a'):
                return 'UA'

            elif filename.startswith('ocean_b'):
                return 'UB'

            return None

        df_stim['scene_id'] = df_stim['integrated_stimulus'].apply(extract_scene_from_integrated)

        scene_to_label = dict(zip(df_cond['scene_stimulus'], df_cond['scene_label']))
        df_stim['scene_label'] = df_stim['scene_id'].map(scene_to_label)

        df_stim.to_csv(dst_stim, index=False)

        print(f"  Updated stim file: {dst_stim}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating all experiment lists...")
    print(f"  Subjects : {list(SUBS)}")
    print(f"  Encoding : {ENCODING_EXP_DIR}")
    print(f"  Retrieval: {RETRIEVAL_EXP_DIR}")
    print(f"  Training : {TRAINING_TASK_DIR}")

    create_stimuli_lists(
        subs                    = SUBS,
        encoding_stim_lists_dir = ENCODING_STIM_LISTS_DIR,
        retrieval_stim_lists_dir= RETRIEVAL_STIM_LISTS_DIR,
        item_norming_file       = ITEM_NORMING_FILE,
        memorability_file       = MEMORABILITY_FILE, 
        is_meg                  = IS_MEG,
    )

    create_training_cond_lists(
        subs                    = SUBS,
        encoding_stim_lists_dir = ENCODING_STIM_LISTS_DIR,
        training_cond_lists_dir = TRAINING_COND_LISTS_DIR,
    )

    create_training_vis_and_stim_lists(
        subs                    = SUBS,
        training_cond_lists_dir = TRAINING_COND_LISTS_DIR,
        training_vis_lists_dir  = TRAINING_VIS_LISTS_DIR,
        training_stim_lists_dir = TRAINING_STIM_LISTS_DIR,
    )


if __name__ == '__main__':
    main()
