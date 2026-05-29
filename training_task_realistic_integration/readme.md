Training Task – Scene Integration Experiment

This folder contains the current working PsychoPy implementation of the training task 
(Parts 3 and 4). The encoding task has not yet been integrated.

MAIN EXPERIMENT FILE
--------------------
Run the experiment using:

    TEST_training_task_CURRENT.psyexp

This file should run directly in PsychoPy when the folder structure is preserved.


FOLDER STRUCTURE
----------------

TEST/
│
├── TEST_training_task_CURRENT.psyexp
│   Main PsychoPy Builder file for the training task.
│
├── data/
│   Output data files from experiment runs.
│
├── backups/
│   Backup versions of earlier .psyexp files and scripts kept for safety.
│   These are not used by the current experiment.
│
├── TESTstimuli/
│   Contains all stimulus materials and generated condition lists.
│
│   ├── cond_lists/
│   │   Scene–name mappings generated per participant.
│   │   Example: scene_names_sub_233.csv
│   │
│   ├── vis_lists/
│   │   Visualization question lists used in Part 3.
│   │
│   ├── stim_lists/
│   │   Stimulus lists used in Part 4 (integrated image training trials).
│   │
│   ├── items/
│   │   Item-only images used during recall.
│   │
│   └── TESTscenes/
│       Background scenes used throughout the training task.
│
│       └── p1_scenes/
│           Background scenes used during Part 1 scene description phase.
│
├── conditions_part2.xlsx
│   Condition file used for Part 2 loops.
│
├── groups.xlsx
│   Defines stimulus group assignments.
│
├── scenes_group1-4.xlsx
│   Scene grouping files used for loop logic.
│
├── CURRENT_cond_lists_generator.py
│   Script used to generate per-participant scene–name condition lists
│   (participants 233–296).
│
├── CURRENT_vis_generator.py
│   Script used to generate visualization lists for Part 3.
│
└── CURRENT_stim_lists_generator.py
    EScript used to generate stim lists for Part 4.


STIMULUS GENERATION
-------------------

Condition lists and visualization lists were generated for participants:

    233–296

using:

    CURRENT_cond_lists_generator.py
    CURRENT_vis_generator.py


EXPERIMENT STATUS
-----------------

Currently implemented and working:

✓ Part 1 – Scene descriptions  
✓ Part 3 – Visualization task  
✓ Part 4 – Integrated image training task  

Not yet implemented:

• Encoding task  
• Retrieval task


RUNNING THE EXPERIMENT
----------------------

1. Download the entire folder.
2. Unzip it.
3. Open:

       TEST_training_task_CURRENT.psyexp

   in PsychoPy Builder.
4. Run the experiment.


NOTES
-----

• File paths are relative to preserve portability across machines.
• The folder structure must remain unchanged for stimuli to load correctly.
