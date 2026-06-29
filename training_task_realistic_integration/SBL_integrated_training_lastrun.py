#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Fri Jun 26 10:51:24 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'TEST_training_task'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': "''",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1470, 956]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = "data/%(participant)s_%(date)s"
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/oliviayin/Desktop/sbl_realistic integration/training_task_realistic_integration/SBL_integrated_training_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions_overall" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='Welcome to the items and scenes memory experiment!\n\nIn this first task, you will learn the names and visual details of four scenes.\n\nYou will then practice visualizing the scenes in detail.\n\nFinally, you will have a short practice run of the next tasks which you will undertake later in the study session.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from init_counterbalance
    
    
    import pandas as pd
    import random
    
    count = 0
    skipTrial = False
    
    cond_file = f"stimuli/cond_lists/scene_names_sub_{expInfo['participant']}.csv"
    df = pd.read_csv(cond_file)
    
    scene_JA_lbl = df.loc[df['scene_stimulus']=='JA','scene_label'].values[0]
    scene_JB_lbl = df.loc[df['scene_stimulus']=='JB','scene_label'].values[0]
    scene_UA_lbl = df.loc[df['scene_stimulus']=='UA','scene_label'].values[0]
    scene_UB_lbl = df.loc[df['scene_stimulus']=='UB','scene_label'].values[0]
    
    
    
    # --- Initialize components for Routine "scene_label_2" ---
    
    # --- Initialize components for Routine "p1_instruction" ---
    txt_insturctions_p1 = visual.TextStim(win=win, name='txt_insturctions_p1',
        text='Part 1.\nIn this part, you will learn about four scenes.\n\nPlease pay attention to the name and to the visual details of each scene. There are three images for each scene, so focus on the consistent characteristics of each scene. The name will appear above the image.\n\nPress SPACEBAR to go to the next scene after you have fully learned the current scene.\n\nYou will be asked to name and visualize them in detail later on.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1" ---
    # Run 'Begin Experiment' code from code_present_scene_name
    templates = {
    'JA': "{name} is a bright tropical rainforest with large buttress-root trees and warm sunlight filtering through the canopy. Visible sunbeams create strong light–shadow contrast on a leaf-covered forest floor, and the space feels open at ground level with sturdy trunks, broad green foliage, and large red and yellow flowers.",
    
    'JB': "{name} is a cool, mist-filled cloud jungle with soft, diffuse lighting and low visibility. Tall trees and the ground are heavily covered in thick green moss, with moisture clinging to surfaces. The damp, foggy air mutes colors and reduces contrast, creating a consistently hazy and subdued visual environment.",
    
    'UA': "{name} is a bright, shallow ocean scene with clear blue water and strong sunlight streaming down in visible beams. The seafloor is mostly pale, rippled sand with a few large smooth rocks where small coral clusters grow. The space feels open with high visibility and a warm tropical atmosphere.",
    
    'UB': "{name} is a cool-toned ocean scene with a riverbed of smooth, rounded pebbles and larger stones rather than sand. The water is slightly hazy with soft, diffuse lighting and no visible sunbeams. Thick green-yellow aquatic plants and muted bluish tones create a more enclosed, freshwater-like atmosphere."
    }
    trial_image = visual.ImageStim(
        win=win,
        name='trial_image', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -.06), draggable=False, size=(None, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    scene_name_part1 = visual.TextStim(win=win, name='scene_name_part1',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.04, wrapWidth=1.4, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_scenename = visual.TextStim(win=win, name='text_scenename',
        text='',
        font='Open Sans',
        pos=(0, .35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    txt_memorize = visual.TextStim(win=win, name='txt_memorize',
        text='Try to memorize the name and the visual details of the scene.',
        font='Open Sans',
        pos=(0, -0.44), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    spaceKey = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_2_instructions" ---
    text_p1_2_instructions = visual.TextStim(win=win, name='text_p1_2_instructions',
        text='You will now see all three images for each scene on one screen. Focus on the similar characteristics consistent across images.\n\nWhen you are done, press the SPACEBAR to continue.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_p1_2_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_2_showALL_2" ---
    image_left = visual.ImageStim(
        win=win,
        name='image_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.38, 0.2), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_center = visual.ImageStim(
        win=win,
        name='image_center', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.25), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image_right = visual.ImageStim(
        win=win,
        name='image_right', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.38, 0.2), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    key_p1_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    txt_scenename_showALL = visual.TextStim(win=win, name='txt_scenename_showALL',
        text='',
        font='Arial',
        pos=(0, 0.43), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "p1_3_instructions_wide" ---
    text_instructions_p1_3 = visual.TextStim(win=win, name='text_instructions_p1_3',
        text='Now you will view all four scenes at once to compare them. These are the wide-shot views of each.\n\nPlease pay attention to the name and to the visual details of each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.053, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_compare" ---
    img1 = visual.ImageStim(
        win=win,
        name='img1', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, .20), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    img2 = visual.ImageStim(
        win=win,
        name='img2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, .2), draggable=False, size=(0.6, 0.3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img3 = visual.ImageStim(
        win=win,
        name='img3', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    img4 = visual.ImageStim(
        win=win,
        name='img4', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(0.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    label1 = visual.TextStim(win=win, name='label1',
        text=None,
        font='Arial',
        pos=(-.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    label2 = visual.TextStim(win=win, name='label2',
        text='Any text\n\nincluding line breaks',
        font='Arial',
        pos=(.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    label3 = visual.TextStim(win=win, name='label3',
        text=None,
        font='Arial',
        pos=(-.35, -.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    label4 = visual.TextStim(win=win, name='label4',
        text=None,
        font='Arial',
        pos=(.35, -0.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    key_p1_compare = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_3_instructions_close" ---
    text_instructions_p1 = visual.TextStim(win=win, name='text_instructions_p1',
        text='Now you will view all four scenes at once to compare them. These are the close-up views of each.\n\nPlease pay attention to the name and to the visual details of each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.053, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_compare_2" ---
    img1_2 = visual.ImageStim(
        win=win,
        name='img1_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, .20), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    img2_2 = visual.ImageStim(
        win=win,
        name='img2_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, .2), draggable=False, size=(0.6, 0.3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img3_2 = visual.ImageStim(
        win=win,
        name='img3_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    img4_2 = visual.ImageStim(
        win=win,
        name='img4_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(0.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    label1_2 = visual.TextStim(win=win, name='label1_2',
        text=None,
        font='Arial',
        pos=(-.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    label2_2 = visual.TextStim(win=win, name='label2_2',
        text='Any text\n\nincluding line breaks',
        font='Arial',
        pos=(.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    label3_2 = visual.TextStim(win=win, name='label3_2',
        text=None,
        font='Arial',
        pos=(-.35, -.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    label4_2 = visual.TextStim(win=win, name='label4_2',
        text=None,
        font='Arial',
        pos=(.35, -0.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    key_p1_compare_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p2_instructions" ---
    text_instructions_part2 = visual.TextStim(win=win, name='text_instructions_part2',
        text='Part 2.\n\nIn this part, you will be asked to choose the correct name for each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.053, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p2" ---
    scene_part2 = visual.ImageStim(
        win=win,
        name='scene_part2', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(960, 540),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "part2_question" ---
    text_part2_question = visual.TextStim(win=win, name='text_part2_question',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_part2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_part2
    total_acc = 0
    logic_vec = [False]*4
    
    
    # --- Initialize components for Routine "part2_feedback" ---
    # Run 'Begin Experiment' code from code_part2_feedback
    msg="doh!"
    text_part2_feedback = visual.TextStim(win=win, name='text_part2_feedback',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "part3_instructions_2" ---
    text_part3_instructions_2 = visual.TextStim(win=win, name='text_part3_instructions_2',
        text='Part 3. \n\nIn this part, you will practice visualizing the scenes in detail.\n\nPress SPACEBAR to continue.\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.053, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_part3_instructions_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p3_reset_vis_vars" ---
    # Run 'Begin Experiment' code from code_5
    count = 0
    logic_vec_ans = [False]*4
    
    # --- Initialize components for Routine "p3_skip_answered_questions" ---
    
    # --- Initialize components for Routine "p3_blank1000" ---
    text = visual.TextStim(win=win, name='text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "p3_visualize" ---
    text_p3_visualize = visual.TextStim(win=win, name='text_p3_visualize',
        text='Please visualize the scene in as much detail as possible',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    textbox_scene_name_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_scene_name_2',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "p3_rate" ---
    text_rate_2 = visual.TextStim(win=win, name='text_rate_2',
        text='How vivid was your visualization?\n\nAnswer the question by pressing "1", "2", "3", or "4"\n\n1. could not visualize\n2. some visualization\n3. good visualization\n4. vivid visualization\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_rate_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p3_question" ---
    text_question_2 = visual.TextStim(win=win, name='text_question_2',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_option_2 = visual.TextStim(win=win, name='text_option_2',
        text='',
        font='Open Sans',
        pos=(0, -.18), draggable=False, height=0.045, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_question_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_5 = visual.TextStim(win=win, name='text_5',
        text='Choose the correct answer',
        font='Open Sans',
        pos=(0, .2), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "p3_study_again" ---
    study_scene_left = visual.ImageStim(
        win=win,
        name='study_scene_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.40, 0), draggable=False, size=(0.7, 0.394),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    study_scene_right = visual.ImageStim(
        win=win,
        name='study_scene_right', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.40, 0), draggable=False, size=(0.7, 0.394),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    study_name1_2 = visual.TextStim(win=win, name='study_name1_2',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_sceneLearn = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_6 = visual.TextStim(win=win, name='text_6',
        text="Press SPACEBAR when you've finished studying",
        font='Open Sans',
        pos=(0, -.38), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='Study the scene again',
        font='Open Sans',
        pos=(0, .45), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "p3_correct_msg" ---
    text_next_scene_2 = visual.TextStim(win=win, name='text_next_scene_2',
        text="Good! \n\nLet's move to the next scene.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "p4_instructions" ---
    text_part4_instructions = visual.TextStim(win=win, name='text_part4_instructions',
        text='Part 4.\n\nNow that you have learned to name and imagine the four scenes in detail, you will have a short practice run in the two tasks that you will undertake soon.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.052, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_part4_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "enc_instruction_screen" ---
    text_enc_instruction = visual.TextStim(win=win, name='text_enc_instruction',
        text='In the first task, you will see different items (objects or animals) realistically integrated into the jungle and undersea scenes you memorized just now. After each presentation of an item and a scene, your goal will be to determine whether they were related or unrelated. \n\nAn item is related to a scene if you think it is plausible that such an animal would live or that such an object would be used in such a scene in real life. If not, they are unrelated.\n\nPress SPACEBAR to continue',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_proceed = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "enc_inst_2" ---
    # Run 'Begin Experiment' code from code_11
    key_related = 'p'
    key_unrelated = 'q'
    
    instruction_text = (
        "If they are related, press P\n"
        "If they are unrelated, press Q\n\n"
        "Press SPACEBAR to continue."
    )
    
    text_relatedness_keys = visual.TextStim(win=win, name='text_relatedness_keys',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_proceed_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "get_stim_list" ---
    # Run 'Begin Experiment' code from code_get_stim_list
    subnum = expInfo['participant']
    stim_list = "stimuli/stim_lists/stimuli_list_training_sub_" + subnum + ".csv"
    
    # --- Initialize components for Routine "integrated_scene_presentation" ---
    image_integrated = visual.ImageStim(
        win=win,
        name='image_integrated', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.05), draggable=False, size=(1.2, 0.675),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "congruency_response" ---
    text_congruency = visual.TextStim(win=win, name='text_congruency',
        text='Related \nor Unrelated?',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_congruency = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank1000_2" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "ret_inst_gen" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='Great job!\n\nNow you will learn about the memory task. It is a bit more complicated than the previous task, with a few changing screens. Please read carefully the detailed instructions before starting the practice run.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_continue_7 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_1" ---
    
    # --- Initialize components for Routine "ret_inst_2" ---
    image_inst2 = visual.ImageStim(
        win=win,
        name='image_inst2', units='height', 
        image='stimuli/instructions_imgs/inst1_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_3" ---
    image_inst3 = visual.ImageStim(
        win=win,
        name='image_inst3', units='height', 
        image='stimuli/instructions_imgs/inst2_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_4" ---
    image_inst4 = visual.ImageStim(
        win=win,
        name='image_inst4', units='height', 
        image='stimuli/instructions_imgs/inst3_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_5" ---
    image_inst5 = visual.ImageStim(
        win=win,
        name='image_inst5', units='height', 
        image='stimuli/instructions_imgs/inst4_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue_5 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_6" ---
    image_inst6 = visual.ImageStim(
        win=win,
        name='image_inst6', units='height', 
        image='stimuli/instructions_imgs/inst5_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue_6 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst_7" ---
    image_inst7 = visual.ImageStim(
        win=win,
        name='image_inst7', units='height', 
        image='stimuli/instructions_imgs/inst6_integrated.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2.11, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_continue = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "item_imagine_scene" ---
    text_imagine = visual.TextStim(win=win, name='text_imagine',
        text='Imagine the associated scene',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    item_cue = visual.ImageStim(
        win=win,
        name='item_cue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.2, 0.675),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "p4_blank1000" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "select_context" ---
    text_contrext_headline = visual.TextStim(win=win, name='text_contrext_headline',
        text='Which Type of scene did you imagine?',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_sea = visual.TextStim(win=win, name='text_sea',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_jungle = visual.TextStim(win=win, name='text_jungle',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_context = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank500" ---
    text_blank = visual.TextStim(win=win, name='text_blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "context_confidence" ---
    text_context_choice = visual.TextStim(win=win, name='text_context_choice',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_context_conf = visual.Slider(win=win, name='slider_context_conf',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=('1.Not confident','2.Somewhat confident','3.Very confident'), ticks=(1, 2, 3), granularity=0.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    key_resp_context_conf = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p4_blank1000" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "select_scene_name" ---
    text_scene_headline = visual.TextStim(win=win, name='text_scene_headline',
        text='The type was',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_actual_context = visual.TextStim(win=win, name='text_actual_context',
        text='',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_q_scene = visual.TextStim(win=win, name='text_q_scene',
        text='Which specific scene was it?',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_labels = visual.TextStim(win=win, name='text_labels',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_scenes = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank500" ---
    text_blank = visual.TextStim(win=win, name='text_blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "select_confidence" ---
    text_chosen_scene = visual.TextStim(win=win, name='text_chosen_scene',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_scene_conf = visual.Slider(win=win, name='slider_scene_conf',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=('1.Not confident','2.Somewhat confident','3.Very confident'), ticks=(1, 2, 3), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    key_resp_scene_conf = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank_3000" ---
    text_blank3000 = visual.TextStim(win=win, name='text_blank3000',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end_screen" ---
    text_end_screen = visual.TextStim(win=win, name='text_end_screen',
        text="Great job!\n\nYou're done with the training.\nSoon we're going to start the two tasks.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_exit = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions_overall" ---
    # create an object to store info about Routine instructions_overall
    instructions_overall = data.Routine(
        name='instructions_overall',
        components=[text_instructions, key_welcome],
    )
    instructions_overall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    # store start times for instructions_overall
    instructions_overall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_overall.tStart = globalClock.getTime(format='float')
    instructions_overall.status = STARTED
    thisExp.addData('instructions_overall.started', instructions_overall.tStart)
    instructions_overall.maxDuration = None
    # keep track of which components have finished
    instructions_overallComponents = instructions_overall.components
    for thisComponent in instructions_overall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_overall" ---
    thisExp.currentRoutine = instructions_overall
    instructions_overall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions* updates
        
        # if text_instructions is starting this frame...
        if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions.frameNStart = frameN  # exact frame index
            text_instructions.tStart = t  # local t and not account for scr refresh
            text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions.started')
            # update status
            text_instructions.status = STARTED
            text_instructions.setAutoDraw(True)
        
        # if text_instructions is active this frame...
        if text_instructions.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_welcome.started')
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_overall,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_overall.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_overall.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_overall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_overall" ---
    for thisComponent in instructions_overall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_overall
    instructions_overall.tStop = globalClock.getTime(format='float')
    instructions_overall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_overall.stopped', instructions_overall.tStop)
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    thisExp.nextEntry()
    # the Routine "instructions_overall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    scene_label_loader = data.TrialHandler2(
        name='scene_label_loader',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions("stimuli/cond_lists/scene_names_sub_%s.csv" % expInfo['participant']), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(scene_label_loader)  # add the loop to the experiment
    thisScene_label_loader = scene_label_loader.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisScene_label_loader.rgb)
    if thisScene_label_loader != None:
        for paramName in thisScene_label_loader:
            globals()[paramName] = thisScene_label_loader[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisScene_label_loader in scene_label_loader:
        scene_label_loader.status = STARTED
        if hasattr(thisScene_label_loader, 'status'):
            thisScene_label_loader.status = STARTED
        currentLoop = scene_label_loader
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisScene_label_loader.rgb)
        if thisScene_label_loader != None:
            for paramName in thisScene_label_loader:
                globals()[paramName] = thisScene_label_loader[paramName]
        
        # --- Prepare to start Routine "scene_label_2" ---
        # create an object to store info about Routine scene_label_2
        scene_label_2 = data.Routine(
            name='scene_label_2',
            components=[],
        )
        scene_label_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_12
        # create dictionaries once (first row)
        if scene_label_loader.thisN == 0:
            scene_label_map = {}
            scene_img_map = {}
        
        # store mapping from the cond_list
        scene_label_map[scene_stimulus] = scene_label
        scene_img_map[scene_stimulus] = scene_img
        # store start times for scene_label_2
        scene_label_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        scene_label_2.tStart = globalClock.getTime(format='float')
        scene_label_2.status = STARTED
        thisExp.addData('scene_label_2.started', scene_label_2.tStart)
        scene_label_2.maxDuration = None
        # keep track of which components have finished
        scene_label_2Components = scene_label_2.components
        for thisComponent in scene_label_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "scene_label_2" ---
        thisExp.currentRoutine = scene_label_2
        scene_label_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisScene_label_loader, 'status') and thisScene_label_loader.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=scene_label_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                scene_label_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if scene_label_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in scene_label_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "scene_label_2" ---
        for thisComponent in scene_label_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for scene_label_2
        scene_label_2.tStop = globalClock.getTime(format='float')
        scene_label_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('scene_label_2.stopped', scene_label_2.tStop)
        # the Routine "scene_label_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisScene_label_loader as finished
        if hasattr(thisScene_label_loader, 'status'):
            thisScene_label_loader.status = FINISHED
        # if awaiting a pause, pause now
        if scene_label_loader.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            scene_label_loader.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'scene_label_loader'
    scene_label_loader.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p1_instruction" ---
    # create an object to store info about Routine p1_instruction
    p1_instruction = data.Routine(
        name='p1_instruction',
        components=[txt_insturctions_p1, key_instructions_p1],
    )
    p1_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1
    key_instructions_p1.keys = []
    key_instructions_p1.rt = []
    _key_instructions_p1_allKeys = []
    # store start times for p1_instruction
    p1_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_instruction.tStart = globalClock.getTime(format='float')
    p1_instruction.status = STARTED
    thisExp.addData('p1_instruction.started', p1_instruction.tStart)
    p1_instruction.maxDuration = None
    # keep track of which components have finished
    p1_instructionComponents = p1_instruction.components
    for thisComponent in p1_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_instruction" ---
    thisExp.currentRoutine = p1_instruction
    p1_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *txt_insturctions_p1* updates
        
        # if txt_insturctions_p1 is starting this frame...
        if txt_insturctions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            txt_insturctions_p1.frameNStart = frameN  # exact frame index
            txt_insturctions_p1.tStart = t  # local t and not account for scr refresh
            txt_insturctions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(txt_insturctions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'txt_insturctions_p1.started')
            # update status
            txt_insturctions_p1.status = STARTED
            txt_insturctions_p1.setAutoDraw(True)
        
        # if txt_insturctions_p1 is active this frame...
        if txt_insturctions_p1.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1* updates
        waitOnFlip = False
        
        # if key_instructions_p1 is starting this frame...
        if key_instructions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1.frameNStart = frameN  # exact frame index
            key_instructions_p1.tStart = t  # local t and not account for scr refresh
            key_instructions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1.started')
            # update status
            key_instructions_p1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_allKeys):
                key_instructions_p1.keys = _key_instructions_p1_allKeys[-1].name  # just the last key pressed
                key_instructions_p1.rt = _key_instructions_p1_allKeys[-1].rt
                key_instructions_p1.duration = _key_instructions_p1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_instruction,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_instruction.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_instruction.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_instruction" ---
    for thisComponent in p1_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_instruction
    p1_instruction.tStop = globalClock.getTime(format='float')
    p1_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_instruction.stopped', p1_instruction.tStop)
    # check responses
    if key_instructions_p1.keys in ['', [], None]:  # No response was made
        key_instructions_p1.keys = None
    thisExp.addData('key_instructions_p1.keys',key_instructions_p1.keys)
    if key_instructions_p1.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1.rt', key_instructions_p1.rt)
        thisExp.addData('key_instructions_p1.duration', key_instructions_p1.duration)
    thisExp.nextEntry()
    # the Routine "p1_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    groups_loop = data.TrialHandler2(
        name='groups_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('groups.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(groups_loop)  # add the loop to the experiment
    thisGroups_loop = groups_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop.rgb)
    if thisGroups_loop != None:
        for paramName in thisGroups_loop:
            globals()[paramName] = thisGroups_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGroups_loop in groups_loop:
        groups_loop.status = STARTED
        if hasattr(thisGroups_loop, 'status'):
            thisGroups_loop.status = STARTED
        currentLoop = groups_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop.rgb)
        if thisGroups_loop != None:
            for paramName in thisGroups_loop:
                globals()[paramName] = thisGroups_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_loop = data.TrialHandler2(
            name='trials_loop',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(conditionFile), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials_loop)  # add the loop to the experiment
        thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_loop in trials_loop:
            trials_loop.status = STARTED
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = STARTED
            currentLoop = trials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
            if thisTrials_loop != None:
                for paramName in thisTrials_loop:
                    globals()[paramName] = thisTrials_loop[paramName]
            
            # --- Prepare to start Routine "p1" ---
            # create an object to store info about Routine p1
            p1 = data.Routine(
                name='p1',
                components=[trial_image, scene_name_part1, text_scenename, txt_memorize, spaceKey],
            )
            p1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_present_scene_name
            filename = os.path.basename(scene_image)
            parts = filename.split('_')
            
            scene_code = parts[1]
            scene_num  = parts[2].split('.')[0]
            
            name = scene_label_map[scene_code]
            
            # --- DESCRIPTION: only on first image ---
            if scene_num == '1':
                txt_scene_present = templates[scene_code].format(name=name)
            else:
                txt_scene_present = ""
            
            # --- NAME: only on second and third images ---
            if scene_num in ['2', '3']:
                txt_scenename = name.upper()
            else:
                txt_scenename = ""
            trial_image.setImage(scene_image)
            scene_name_part1.setPos((0, .34))
            scene_name_part1.setText(txt_scene_present)
            text_scenename.setText(txt_scenename)
            # create starting attributes for spaceKey
            spaceKey.keys = []
            spaceKey.rt = []
            _spaceKey_allKeys = []
            # store start times for p1
            p1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p1.tStart = globalClock.getTime(format='float')
            p1.status = STARTED
            thisExp.addData('p1.started', p1.tStart)
            p1.maxDuration = None
            # keep track of which components have finished
            p1Components = p1.components
            for thisComponent in p1.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p1" ---
            thisExp.currentRoutine = p1
            p1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_image* updates
                
                # if trial_image is starting this frame...
                if trial_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_image.frameNStart = frameN  # exact frame index
                    trial_image.tStart = t  # local t and not account for scr refresh
                    trial_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_image.started')
                    # update status
                    trial_image.status = STARTED
                    trial_image.setAutoDraw(True)
                
                # if trial_image is active this frame...
                if trial_image.status == STARTED:
                    # update params
                    pass
                
                # *scene_name_part1* updates
                
                # if scene_name_part1 is starting this frame...
                if scene_name_part1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scene_name_part1.frameNStart = frameN  # exact frame index
                    scene_name_part1.tStart = t  # local t and not account for scr refresh
                    scene_name_part1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scene_name_part1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'scene_name_part1.started')
                    # update status
                    scene_name_part1.status = STARTED
                    scene_name_part1.setAutoDraw(True)
                
                # if scene_name_part1 is active this frame...
                if scene_name_part1.status == STARTED:
                    # update params
                    pass
                
                # *text_scenename* updates
                
                # if text_scenename is starting this frame...
                if text_scenename.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_scenename.frameNStart = frameN  # exact frame index
                    text_scenename.tStart = t  # local t and not account for scr refresh
                    text_scenename.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_scenename, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_scenename.started')
                    # update status
                    text_scenename.status = STARTED
                    text_scenename.setAutoDraw(True)
                
                # if text_scenename is active this frame...
                if text_scenename.status == STARTED:
                    # update params
                    pass
                
                # *txt_memorize* updates
                
                # if txt_memorize is starting this frame...
                if txt_memorize.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    txt_memorize.frameNStart = frameN  # exact frame index
                    txt_memorize.tStart = t  # local t and not account for scr refresh
                    txt_memorize.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(txt_memorize, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'txt_memorize.started')
                    # update status
                    txt_memorize.status = STARTED
                    txt_memorize.setAutoDraw(True)
                
                # if txt_memorize is active this frame...
                if txt_memorize.status == STARTED:
                    # update params
                    pass
                
                # *spaceKey* updates
                waitOnFlip = False
                
                # if spaceKey is starting this frame...
                if spaceKey.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    spaceKey.frameNStart = frameN  # exact frame index
                    spaceKey.tStart = t  # local t and not account for scr refresh
                    spaceKey.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(spaceKey, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'spaceKey.started')
                    # update status
                    spaceKey.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(spaceKey.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(spaceKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if spaceKey.status == STARTED and not waitOnFlip:
                    theseKeys = spaceKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _spaceKey_allKeys.extend(theseKeys)
                    if len(_spaceKey_allKeys):
                        spaceKey.keys = _spaceKey_allKeys[-1].name  # just the last key pressed
                        spaceKey.rt = _spaceKey_allKeys[-1].rt
                        spaceKey.duration = _spaceKey_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p1,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p1.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p1.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p1" ---
            for thisComponent in p1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p1
            p1.tStop = globalClock.getTime(format='float')
            p1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p1.stopped', p1.tStop)
            # check responses
            if spaceKey.keys in ['', [], None]:  # No response was made
                spaceKey.keys = None
            trials_loop.addData('spaceKey.keys',spaceKey.keys)
            if spaceKey.keys != None:  # we had a response
                trials_loop.addData('spaceKey.rt', spaceKey.rt)
                trials_loop.addData('spaceKey.duration', spaceKey.duration)
            # the Routine "p1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrials_loop as finished
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = FINISHED
            # if awaiting a pause, pause now
            if trials_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_loop'
        trials_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisGroups_loop as finished
        if hasattr(thisGroups_loop, 'status'):
            thisGroups_loop.status = FINISHED
        # if awaiting a pause, pause now
        if groups_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            groups_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'groups_loop'
    groups_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p1_2_instructions" ---
    # create an object to store info about Routine p1_2_instructions
    p1_2_instructions = data.Routine(
        name='p1_2_instructions',
        components=[text_p1_2_instructions, key_p1_2_instructions],
    )
    p1_2_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_p1_2_instructions
    key_p1_2_instructions.keys = []
    key_p1_2_instructions.rt = []
    _key_p1_2_instructions_allKeys = []
    # store start times for p1_2_instructions
    p1_2_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_2_instructions.tStart = globalClock.getTime(format='float')
    p1_2_instructions.status = STARTED
    thisExp.addData('p1_2_instructions.started', p1_2_instructions.tStart)
    p1_2_instructions.maxDuration = None
    # keep track of which components have finished
    p1_2_instructionsComponents = p1_2_instructions.components
    for thisComponent in p1_2_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_2_instructions" ---
    thisExp.currentRoutine = p1_2_instructions
    p1_2_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_p1_2_instructions* updates
        
        # if text_p1_2_instructions is starting this frame...
        if text_p1_2_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_p1_2_instructions.frameNStart = frameN  # exact frame index
            text_p1_2_instructions.tStart = t  # local t and not account for scr refresh
            text_p1_2_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_p1_2_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_p1_2_instructions.started')
            # update status
            text_p1_2_instructions.status = STARTED
            text_p1_2_instructions.setAutoDraw(True)
        
        # if text_p1_2_instructions is active this frame...
        if text_p1_2_instructions.status == STARTED:
            # update params
            pass
        
        # *key_p1_2_instructions* updates
        waitOnFlip = False
        
        # if key_p1_2_instructions is starting this frame...
        if key_p1_2_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_2_instructions.frameNStart = frameN  # exact frame index
            key_p1_2_instructions.tStart = t  # local t and not account for scr refresh
            key_p1_2_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_2_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_2_instructions.started')
            # update status
            key_p1_2_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_2_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_2_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_2_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_2_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_2_instructions_allKeys.extend(theseKeys)
            if len(_key_p1_2_instructions_allKeys):
                key_p1_2_instructions.keys = _key_p1_2_instructions_allKeys[-1].name  # just the last key pressed
                key_p1_2_instructions.rt = _key_p1_2_instructions_allKeys[-1].rt
                key_p1_2_instructions.duration = _key_p1_2_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_2_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_2_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_2_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_2_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_2_instructions" ---
    for thisComponent in p1_2_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_2_instructions
    p1_2_instructions.tStop = globalClock.getTime(format='float')
    p1_2_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_2_instructions.stopped', p1_2_instructions.tStop)
    # check responses
    if key_p1_2_instructions.keys in ['', [], None]:  # No response was made
        key_p1_2_instructions.keys = None
    thisExp.addData('key_p1_2_instructions.keys',key_p1_2_instructions.keys)
    if key_p1_2_instructions.keys != None:  # we had a response
        thisExp.addData('key_p1_2_instructions.rt', key_p1_2_instructions.rt)
        thisExp.addData('key_p1_2_instructions.duration', key_p1_2_instructions.duration)
    thisExp.nextEntry()
    # the Routine "p1_2_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    groups_loop_p1_2 = data.TrialHandler2(
        name='groups_loop_p1_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions_part2.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(groups_loop_p1_2)  # add the loop to the experiment
    thisGroups_loop_p1_2 = groups_loop_p1_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop_p1_2.rgb)
    if thisGroups_loop_p1_2 != None:
        for paramName in thisGroups_loop_p1_2:
            globals()[paramName] = thisGroups_loop_p1_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGroups_loop_p1_2 in groups_loop_p1_2:
        groups_loop_p1_2.status = STARTED
        if hasattr(thisGroups_loop_p1_2, 'status'):
            thisGroups_loop_p1_2.status = STARTED
        currentLoop = groups_loop_p1_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop_p1_2.rgb)
        if thisGroups_loop_p1_2 != None:
            for paramName in thisGroups_loop_p1_2:
                globals()[paramName] = thisGroups_loop_p1_2[paramName]
        
        # --- Prepare to start Routine "p1_2_showALL_2" ---
        # create an object to store info about Routine p1_2_showALL_2
        p1_2_showALL_2 = data.Routine(
            name='p1_2_showALL_2',
            components=[image_left, image_center, image_right, key_p1_2, txt_scenename_showALL],
        )
        p1_2_showALL_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_left.setImage(Image1)
        image_center.setImage(Image2)
        image_right.setImage(Image3)
        # create starting attributes for key_p1_2
        key_p1_2.keys = []
        key_p1_2.rt = []
        _key_p1_2_allKeys = []
        txt_scenename_showALL.setText(eval(scene_name_showALL))
        # store start times for p1_2_showALL_2
        p1_2_showALL_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p1_2_showALL_2.tStart = globalClock.getTime(format='float')
        p1_2_showALL_2.status = STARTED
        thisExp.addData('p1_2_showALL_2.started', p1_2_showALL_2.tStart)
        p1_2_showALL_2.maxDuration = None
        # keep track of which components have finished
        p1_2_showALL_2Components = p1_2_showALL_2.components
        for thisComponent in p1_2_showALL_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p1_2_showALL_2" ---
        thisExp.currentRoutine = p1_2_showALL_2
        p1_2_showALL_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisGroups_loop_p1_2, 'status') and thisGroups_loop_p1_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_left* updates
            
            # if image_left is starting this frame...
            if image_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_left.frameNStart = frameN  # exact frame index
                image_left.tStart = t  # local t and not account for scr refresh
                image_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_left.started')
                # update status
                image_left.status = STARTED
                image_left.setAutoDraw(True)
            
            # if image_left is active this frame...
            if image_left.status == STARTED:
                # update params
                pass
            
            # *image_center* updates
            
            # if image_center is starting this frame...
            if image_center.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_center.frameNStart = frameN  # exact frame index
                image_center.tStart = t  # local t and not account for scr refresh
                image_center.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_center, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_center.started')
                # update status
                image_center.status = STARTED
                image_center.setAutoDraw(True)
            
            # if image_center is active this frame...
            if image_center.status == STARTED:
                # update params
                pass
            
            # *image_right* updates
            
            # if image_right is starting this frame...
            if image_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_right.frameNStart = frameN  # exact frame index
                image_right.tStart = t  # local t and not account for scr refresh
                image_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_right.started')
                # update status
                image_right.status = STARTED
                image_right.setAutoDraw(True)
            
            # if image_right is active this frame...
            if image_right.status == STARTED:
                # update params
                pass
            
            # *key_p1_2* updates
            waitOnFlip = False
            
            # if key_p1_2 is starting this frame...
            if key_p1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_p1_2.frameNStart = frameN  # exact frame index
                key_p1_2.tStart = t  # local t and not account for scr refresh
                key_p1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_p1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_p1_2.started')
                # update status
                key_p1_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_p1_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_p1_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_p1_2.status == STARTED and not waitOnFlip:
                theseKeys = key_p1_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_p1_2_allKeys.extend(theseKeys)
                if len(_key_p1_2_allKeys):
                    key_p1_2.keys = _key_p1_2_allKeys[-1].name  # just the last key pressed
                    key_p1_2.rt = _key_p1_2_allKeys[-1].rt
                    key_p1_2.duration = _key_p1_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *txt_scenename_showALL* updates
            
            # if txt_scenename_showALL is starting this frame...
            if txt_scenename_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                txt_scenename_showALL.frameNStart = frameN  # exact frame index
                txt_scenename_showALL.tStart = t  # local t and not account for scr refresh
                txt_scenename_showALL.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(txt_scenename_showALL, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'txt_scenename_showALL.started')
                # update status
                txt_scenename_showALL.status = STARTED
                txt_scenename_showALL.setAutoDraw(True)
            
            # if txt_scenename_showALL is active this frame...
            if txt_scenename_showALL.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p1_2_showALL_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p1_2_showALL_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p1_2_showALL_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p1_2_showALL_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p1_2_showALL_2" ---
        for thisComponent in p1_2_showALL_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p1_2_showALL_2
        p1_2_showALL_2.tStop = globalClock.getTime(format='float')
        p1_2_showALL_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p1_2_showALL_2.stopped', p1_2_showALL_2.tStop)
        # check responses
        if key_p1_2.keys in ['', [], None]:  # No response was made
            key_p1_2.keys = None
        groups_loop_p1_2.addData('key_p1_2.keys',key_p1_2.keys)
        if key_p1_2.keys != None:  # we had a response
            groups_loop_p1_2.addData('key_p1_2.rt', key_p1_2.rt)
            groups_loop_p1_2.addData('key_p1_2.duration', key_p1_2.duration)
        # the Routine "p1_2_showALL_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisGroups_loop_p1_2 as finished
        if hasattr(thisGroups_loop_p1_2, 'status'):
            thisGroups_loop_p1_2.status = FINISHED
        # if awaiting a pause, pause now
        if groups_loop_p1_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            groups_loop_p1_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'groups_loop_p1_2'
    groups_loop_p1_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p1_3_instructions_wide" ---
    # create an object to store info about Routine p1_3_instructions_wide
    p1_3_instructions_wide = data.Routine(
        name='p1_3_instructions_wide',
        components=[text_instructions_p1_3, key_instructions_p1_3],
    )
    p1_3_instructions_wide.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1_3
    key_instructions_p1_3.keys = []
    key_instructions_p1_3.rt = []
    _key_instructions_p1_3_allKeys = []
    # store start times for p1_3_instructions_wide
    p1_3_instructions_wide.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_3_instructions_wide.tStart = globalClock.getTime(format='float')
    p1_3_instructions_wide.status = STARTED
    thisExp.addData('p1_3_instructions_wide.started', p1_3_instructions_wide.tStart)
    p1_3_instructions_wide.maxDuration = None
    # keep track of which components have finished
    p1_3_instructions_wideComponents = p1_3_instructions_wide.components
    for thisComponent in p1_3_instructions_wide.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_3_instructions_wide" ---
    thisExp.currentRoutine = p1_3_instructions_wide
    p1_3_instructions_wide.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_p1_3* updates
        
        # if text_instructions_p1_3 is starting this frame...
        if text_instructions_p1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_p1_3.frameNStart = frameN  # exact frame index
            text_instructions_p1_3.tStart = t  # local t and not account for scr refresh
            text_instructions_p1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_p1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_p1_3.started')
            # update status
            text_instructions_p1_3.status = STARTED
            text_instructions_p1_3.setAutoDraw(True)
        
        # if text_instructions_p1_3 is active this frame...
        if text_instructions_p1_3.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1_3* updates
        waitOnFlip = False
        
        # if key_instructions_p1_3 is starting this frame...
        if key_instructions_p1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1_3.frameNStart = frameN  # exact frame index
            key_instructions_p1_3.tStart = t  # local t and not account for scr refresh
            key_instructions_p1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1_3.started')
            # update status
            key_instructions_p1_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_3_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_3_allKeys):
                key_instructions_p1_3.keys = _key_instructions_p1_3_allKeys[-1].name  # just the last key pressed
                key_instructions_p1_3.rt = _key_instructions_p1_3_allKeys[-1].rt
                key_instructions_p1_3.duration = _key_instructions_p1_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_3_instructions_wide,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_3_instructions_wide.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_3_instructions_wide.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_3_instructions_wide.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_3_instructions_wide" ---
    for thisComponent in p1_3_instructions_wide.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_3_instructions_wide
    p1_3_instructions_wide.tStop = globalClock.getTime(format='float')
    p1_3_instructions_wide.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_3_instructions_wide.stopped', p1_3_instructions_wide.tStop)
    # check responses
    if key_instructions_p1_3.keys in ['', [], None]:  # No response was made
        key_instructions_p1_3.keys = None
    thisExp.addData('key_instructions_p1_3.keys',key_instructions_p1_3.keys)
    if key_instructions_p1_3.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1_3.rt', key_instructions_p1_3.rt)
        thisExp.addData('key_instructions_p1_3.duration', key_instructions_p1_3.duration)
    thisExp.nextEntry()
    # the Routine "p1_3_instructions_wide" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_compare" ---
    # create an object to store info about Routine p1_compare
    p1_compare = data.Routine(
        name='p1_compare',
        components=[img1, img2, img3, img4, label1, label2, label3, label4, key_p1_compare],
    )
    p1_compare.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from setStimuli
    
    
    # fixed image paths
    JA_img = "stimuli/scenes/JA_3.jpg"
    JB_img = "stimuli/scenes/JB_2.jpg"
    UA_img = "stimuli/scenes/UA_2.jpg"
    UB_img = "stimuli/scenes/UB_1.jpg"
    
    # labels pulled from condition list variables created in Begin Experiment
    JA_lbl = scene_JA_lbl
    JB_lbl = scene_JB_lbl
    UA_lbl = scene_UA_lbl
    UB_lbl = scene_UB_lbl
    
    # fixed layout
    img1.image = JA_img
    label1.text = JA_lbl
    
    img2.image = JB_img
    label2.text = JB_lbl
    
    img3.image = UA_img
    label3.text = UA_lbl
    
    img4.image = UB_img
    label4.text = UB_lbl
    # create starting attributes for key_p1_compare
    key_p1_compare.keys = []
    key_p1_compare.rt = []
    _key_p1_compare_allKeys = []
    # store start times for p1_compare
    p1_compare.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_compare.tStart = globalClock.getTime(format='float')
    p1_compare.status = STARTED
    thisExp.addData('p1_compare.started', p1_compare.tStart)
    p1_compare.maxDuration = None
    # keep track of which components have finished
    p1_compareComponents = p1_compare.components
    for thisComponent in p1_compare.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_compare" ---
    thisExp.currentRoutine = p1_compare
    p1_compare.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *img1* updates
        
        # if img1 is starting this frame...
        if img1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img1.frameNStart = frameN  # exact frame index
            img1.tStart = t  # local t and not account for scr refresh
            img1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img1.started')
            # update status
            img1.status = STARTED
            img1.setAutoDraw(True)
        
        # if img1 is active this frame...
        if img1.status == STARTED:
            # update params
            pass
        
        # *img2* updates
        
        # if img2 is starting this frame...
        if img2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img2.frameNStart = frameN  # exact frame index
            img2.tStart = t  # local t and not account for scr refresh
            img2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img2.started')
            # update status
            img2.status = STARTED
            img2.setAutoDraw(True)
        
        # if img2 is active this frame...
        if img2.status == STARTED:
            # update params
            pass
        
        # *img3* updates
        
        # if img3 is starting this frame...
        if img3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img3.frameNStart = frameN  # exact frame index
            img3.tStart = t  # local t and not account for scr refresh
            img3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img3.started')
            # update status
            img3.status = STARTED
            img3.setAutoDraw(True)
        
        # if img3 is active this frame...
        if img3.status == STARTED:
            # update params
            pass
        
        # *img4* updates
        
        # if img4 is starting this frame...
        if img4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img4.frameNStart = frameN  # exact frame index
            img4.tStart = t  # local t and not account for scr refresh
            img4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img4.started')
            # update status
            img4.status = STARTED
            img4.setAutoDraw(True)
        
        # if img4 is active this frame...
        if img4.status == STARTED:
            # update params
            pass
        
        # *label1* updates
        
        # if label1 is starting this frame...
        if label1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label1.frameNStart = frameN  # exact frame index
            label1.tStart = t  # local t and not account for scr refresh
            label1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label1.started')
            # update status
            label1.status = STARTED
            label1.setAutoDraw(True)
        
        # if label1 is active this frame...
        if label1.status == STARTED:
            # update params
            pass
        
        # *label2* updates
        
        # if label2 is starting this frame...
        if label2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label2.frameNStart = frameN  # exact frame index
            label2.tStart = t  # local t and not account for scr refresh
            label2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label2.started')
            # update status
            label2.status = STARTED
            label2.setAutoDraw(True)
        
        # if label2 is active this frame...
        if label2.status == STARTED:
            # update params
            pass
        
        # *label3* updates
        
        # if label3 is starting this frame...
        if label3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label3.frameNStart = frameN  # exact frame index
            label3.tStart = t  # local t and not account for scr refresh
            label3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label3.started')
            # update status
            label3.status = STARTED
            label3.setAutoDraw(True)
        
        # if label3 is active this frame...
        if label3.status == STARTED:
            # update params
            pass
        
        # *label4* updates
        
        # if label4 is starting this frame...
        if label4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label4.frameNStart = frameN  # exact frame index
            label4.tStart = t  # local t and not account for scr refresh
            label4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label4.started')
            # update status
            label4.status = STARTED
            label4.setAutoDraw(True)
        
        # if label4 is active this frame...
        if label4.status == STARTED:
            # update params
            pass
        
        # *key_p1_compare* updates
        waitOnFlip = False
        
        # if key_p1_compare is starting this frame...
        if key_p1_compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_compare.frameNStart = frameN  # exact frame index
            key_p1_compare.tStart = t  # local t and not account for scr refresh
            key_p1_compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_compare, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_compare.started')
            # update status
            key_p1_compare.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_compare.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_compare.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_compare.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_compare.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_compare_allKeys.extend(theseKeys)
            if len(_key_p1_compare_allKeys):
                key_p1_compare.keys = _key_p1_compare_allKeys[-1].name  # just the last key pressed
                key_p1_compare.rt = _key_p1_compare_allKeys[-1].rt
                key_p1_compare.duration = _key_p1_compare_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_compare,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_compare.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_compare.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_compare.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_compare" ---
    for thisComponent in p1_compare.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_compare
    p1_compare.tStop = globalClock.getTime(format='float')
    p1_compare.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_compare.stopped', p1_compare.tStop)
    # check responses
    if key_p1_compare.keys in ['', [], None]:  # No response was made
        key_p1_compare.keys = None
    thisExp.addData('key_p1_compare.keys',key_p1_compare.keys)
    if key_p1_compare.keys != None:  # we had a response
        thisExp.addData('key_p1_compare.rt', key_p1_compare.rt)
        thisExp.addData('key_p1_compare.duration', key_p1_compare.duration)
    thisExp.nextEntry()
    # the Routine "p1_compare" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_3_instructions_close" ---
    # create an object to store info about Routine p1_3_instructions_close
    p1_3_instructions_close = data.Routine(
        name='p1_3_instructions_close',
        components=[text_instructions_p1, key_instructions_p1_4],
    )
    p1_3_instructions_close.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1_4
    key_instructions_p1_4.keys = []
    key_instructions_p1_4.rt = []
    _key_instructions_p1_4_allKeys = []
    # store start times for p1_3_instructions_close
    p1_3_instructions_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_3_instructions_close.tStart = globalClock.getTime(format='float')
    p1_3_instructions_close.status = STARTED
    thisExp.addData('p1_3_instructions_close.started', p1_3_instructions_close.tStart)
    p1_3_instructions_close.maxDuration = None
    # keep track of which components have finished
    p1_3_instructions_closeComponents = p1_3_instructions_close.components
    for thisComponent in p1_3_instructions_close.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_3_instructions_close" ---
    thisExp.currentRoutine = p1_3_instructions_close
    p1_3_instructions_close.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_p1* updates
        
        # if text_instructions_p1 is starting this frame...
        if text_instructions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_p1.frameNStart = frameN  # exact frame index
            text_instructions_p1.tStart = t  # local t and not account for scr refresh
            text_instructions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_p1.started')
            # update status
            text_instructions_p1.status = STARTED
            text_instructions_p1.setAutoDraw(True)
        
        # if text_instructions_p1 is active this frame...
        if text_instructions_p1.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1_4* updates
        waitOnFlip = False
        
        # if key_instructions_p1_4 is starting this frame...
        if key_instructions_p1_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1_4.frameNStart = frameN  # exact frame index
            key_instructions_p1_4.tStart = t  # local t and not account for scr refresh
            key_instructions_p1_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1_4.started')
            # update status
            key_instructions_p1_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1_4.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_4_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_4_allKeys):
                key_instructions_p1_4.keys = _key_instructions_p1_4_allKeys[-1].name  # just the last key pressed
                key_instructions_p1_4.rt = _key_instructions_p1_4_allKeys[-1].rt
                key_instructions_p1_4.duration = _key_instructions_p1_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_3_instructions_close,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_3_instructions_close.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_3_instructions_close.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_3_instructions_close.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_3_instructions_close" ---
    for thisComponent in p1_3_instructions_close.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_3_instructions_close
    p1_3_instructions_close.tStop = globalClock.getTime(format='float')
    p1_3_instructions_close.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_3_instructions_close.stopped', p1_3_instructions_close.tStop)
    # check responses
    if key_instructions_p1_4.keys in ['', [], None]:  # No response was made
        key_instructions_p1_4.keys = None
    thisExp.addData('key_instructions_p1_4.keys',key_instructions_p1_4.keys)
    if key_instructions_p1_4.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1_4.rt', key_instructions_p1_4.rt)
        thisExp.addData('key_instructions_p1_4.duration', key_instructions_p1_4.duration)
    thisExp.nextEntry()
    # the Routine "p1_3_instructions_close" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_compare_2" ---
    # create an object to store info about Routine p1_compare_2
    p1_compare_2 = data.Routine(
        name='p1_compare_2',
        components=[img1_2, img2_2, img3_2, img4_2, label1_2, label2_2, label3_2, label4_2, key_p1_compare_2],
    )
    p1_compare_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from setStimuli_2
    
    # fixed image paths
    JA_img = "stimuli/scenes/JA_1.jpg"
    JB_img = "stimuli/scenes/JB_3.jpg"
    UA_img = "stimuli/scenes/UA_1.jpg"
    UB_img = "stimuli/scenes/UB_2.jpg"
    
    # labels pulled from condition list variables created in Begin Experiment
    JA_lbl = scene_JA_lbl
    JB_lbl = scene_JB_lbl
    UA_lbl = scene_UA_lbl
    UB_lbl = scene_UB_lbl
    
    # fixed layout
    img1_2.image = JA_img
    label1_2.text = JA_lbl
    
    img2_2.image = JB_img
    label2_2.text = JB_lbl
    
    img3_2.image = UA_img
    label3_2.text = UA_lbl
    
    img4_2.image = UB_img
    label4_2.text = UB_lbl
    # create starting attributes for key_p1_compare_2
    key_p1_compare_2.keys = []
    key_p1_compare_2.rt = []
    _key_p1_compare_2_allKeys = []
    # store start times for p1_compare_2
    p1_compare_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_compare_2.tStart = globalClock.getTime(format='float')
    p1_compare_2.status = STARTED
    thisExp.addData('p1_compare_2.started', p1_compare_2.tStart)
    p1_compare_2.maxDuration = None
    # keep track of which components have finished
    p1_compare_2Components = p1_compare_2.components
    for thisComponent in p1_compare_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p1_compare_2" ---
    thisExp.currentRoutine = p1_compare_2
    p1_compare_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *img1_2* updates
        
        # if img1_2 is starting this frame...
        if img1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img1_2.frameNStart = frameN  # exact frame index
            img1_2.tStart = t  # local t and not account for scr refresh
            img1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img1_2.started')
            # update status
            img1_2.status = STARTED
            img1_2.setAutoDraw(True)
        
        # if img1_2 is active this frame...
        if img1_2.status == STARTED:
            # update params
            pass
        
        # *img2_2* updates
        
        # if img2_2 is starting this frame...
        if img2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img2_2.frameNStart = frameN  # exact frame index
            img2_2.tStart = t  # local t and not account for scr refresh
            img2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img2_2.started')
            # update status
            img2_2.status = STARTED
            img2_2.setAutoDraw(True)
        
        # if img2_2 is active this frame...
        if img2_2.status == STARTED:
            # update params
            pass
        
        # *img3_2* updates
        
        # if img3_2 is starting this frame...
        if img3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img3_2.frameNStart = frameN  # exact frame index
            img3_2.tStart = t  # local t and not account for scr refresh
            img3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img3_2.started')
            # update status
            img3_2.status = STARTED
            img3_2.setAutoDraw(True)
        
        # if img3_2 is active this frame...
        if img3_2.status == STARTED:
            # update params
            pass
        
        # *img4_2* updates
        
        # if img4_2 is starting this frame...
        if img4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img4_2.frameNStart = frameN  # exact frame index
            img4_2.tStart = t  # local t and not account for scr refresh
            img4_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img4_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img4_2.started')
            # update status
            img4_2.status = STARTED
            img4_2.setAutoDraw(True)
        
        # if img4_2 is active this frame...
        if img4_2.status == STARTED:
            # update params
            pass
        
        # *label1_2* updates
        
        # if label1_2 is starting this frame...
        if label1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label1_2.frameNStart = frameN  # exact frame index
            label1_2.tStart = t  # local t and not account for scr refresh
            label1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label1_2.started')
            # update status
            label1_2.status = STARTED
            label1_2.setAutoDraw(True)
        
        # if label1_2 is active this frame...
        if label1_2.status == STARTED:
            # update params
            pass
        
        # *label2_2* updates
        
        # if label2_2 is starting this frame...
        if label2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label2_2.frameNStart = frameN  # exact frame index
            label2_2.tStart = t  # local t and not account for scr refresh
            label2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label2_2.started')
            # update status
            label2_2.status = STARTED
            label2_2.setAutoDraw(True)
        
        # if label2_2 is active this frame...
        if label2_2.status == STARTED:
            # update params
            pass
        
        # *label3_2* updates
        
        # if label3_2 is starting this frame...
        if label3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label3_2.frameNStart = frameN  # exact frame index
            label3_2.tStart = t  # local t and not account for scr refresh
            label3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label3_2.started')
            # update status
            label3_2.status = STARTED
            label3_2.setAutoDraw(True)
        
        # if label3_2 is active this frame...
        if label3_2.status == STARTED:
            # update params
            pass
        
        # *label4_2* updates
        
        # if label4_2 is starting this frame...
        if label4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label4_2.frameNStart = frameN  # exact frame index
            label4_2.tStart = t  # local t and not account for scr refresh
            label4_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label4_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label4_2.started')
            # update status
            label4_2.status = STARTED
            label4_2.setAutoDraw(True)
        
        # if label4_2 is active this frame...
        if label4_2.status == STARTED:
            # update params
            pass
        
        # *key_p1_compare_2* updates
        waitOnFlip = False
        
        # if key_p1_compare_2 is starting this frame...
        if key_p1_compare_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_compare_2.frameNStart = frameN  # exact frame index
            key_p1_compare_2.tStart = t  # local t and not account for scr refresh
            key_p1_compare_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_compare_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_compare_2.started')
            # update status
            key_p1_compare_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_compare_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_compare_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_compare_2.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_compare_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_compare_2_allKeys.extend(theseKeys)
            if len(_key_p1_compare_2_allKeys):
                key_p1_compare_2.keys = _key_p1_compare_2_allKeys[-1].name  # just the last key pressed
                key_p1_compare_2.rt = _key_p1_compare_2_allKeys[-1].rt
                key_p1_compare_2.duration = _key_p1_compare_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p1_compare_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_compare_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_compare_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_compare_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_compare_2" ---
    for thisComponent in p1_compare_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_compare_2
    p1_compare_2.tStop = globalClock.getTime(format='float')
    p1_compare_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_compare_2.stopped', p1_compare_2.tStop)
    # check responses
    if key_p1_compare_2.keys in ['', [], None]:  # No response was made
        key_p1_compare_2.keys = None
    thisExp.addData('key_p1_compare_2.keys',key_p1_compare_2.keys)
    if key_p1_compare_2.keys != None:  # we had a response
        thisExp.addData('key_p1_compare_2.rt', key_p1_compare_2.rt)
        thisExp.addData('key_p1_compare_2.duration', key_p1_compare_2.duration)
    thisExp.nextEntry()
    # the Routine "p1_compare_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p2_instructions" ---
    # create an object to store info about Routine p2_instructions
    p2_instructions = data.Routine(
        name='p2_instructions',
        components=[text_instructions_part2, key_resp],
    )
    p2_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for p2_instructions
    p2_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p2_instructions.tStart = globalClock.getTime(format='float')
    p2_instructions.status = STARTED
    thisExp.addData('p2_instructions.started', p2_instructions.tStart)
    p2_instructions.maxDuration = None
    # keep track of which components have finished
    p2_instructionsComponents = p2_instructions.components
    for thisComponent in p2_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p2_instructions" ---
    thisExp.currentRoutine = p2_instructions
    p2_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_part2* updates
        
        # if text_instructions_part2 is starting this frame...
        if text_instructions_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_part2.frameNStart = frameN  # exact frame index
            text_instructions_part2.tStart = t  # local t and not account for scr refresh
            text_instructions_part2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_part2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_part2.started')
            # update status
            text_instructions_part2.status = STARTED
            text_instructions_part2.setAutoDraw(True)
        
        # if text_instructions_part2 is active this frame...
        if text_instructions_part2.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p2_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p2_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p2_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p2_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p2_instructions" ---
    for thisComponent in p2_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p2_instructions
    p2_instructions.tStop = globalClock.getTime(format='float')
    p2_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p2_instructions.stopped', p2_instructions.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "p2_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    part2_loop = data.TrialHandler2(
        name='part2_loop',
        nReps=999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('scene_stims_new.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(part2_loop)  # add the loop to the experiment
    thisPart2_loop = part2_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPart2_loop.rgb)
    if thisPart2_loop != None:
        for paramName in thisPart2_loop:
            globals()[paramName] = thisPart2_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPart2_loop in part2_loop:
        part2_loop.status = STARTED
        if hasattr(thisPart2_loop, 'status'):
            thisPart2_loop.status = STARTED
        currentLoop = part2_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPart2_loop.rgb)
        if thisPart2_loop != None:
            for paramName in thisPart2_loop:
                globals()[paramName] = thisPart2_loop[paramName]
        
        # --- Prepare to start Routine "p2" ---
        # create an object to store info about Routine p2
        p2 = data.Routine(
            name='p2',
            components=[scene_part2],
        )
        p2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        # Randomly pick which image variant (1, 2, or 3) to show
        image_variant = random.choice([1, 2, 3])
        
        # Build the filename based on the correct scene
        scene_codes = ['JA', 'JB', 'UA', 'UB']
        current_scene = scene_codes[int(correct_key) - 1]
        
        # Use .jpg extension and include the folder path
        current_image = f'stimuli/scenes/{current_scene}_{image_variant}.jpg'
        scene_part2.setImage(current_image)
        # store start times for p2
        p2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p2.tStart = globalClock.getTime(format='float')
        p2.status = STARTED
        thisExp.addData('p2.started', p2.tStart)
        p2.maxDuration = None
        # keep track of which components have finished
        p2Components = p2.components
        for thisComponent in p2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p2" ---
        thisExp.currentRoutine = p2
        p2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scene_part2* updates
            
            # if scene_part2 is starting this frame...
            if scene_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scene_part2.frameNStart = frameN  # exact frame index
                scene_part2.tStart = t  # local t and not account for scr refresh
                scene_part2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scene_part2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scene_part2.started')
                # update status
                scene_part2.status = STARTED
                scene_part2.setAutoDraw(True)
            
            # if scene_part2 is active this frame...
            if scene_part2.status == STARTED:
                # update params
                pass
            
            # if scene_part2 is stopping this frame...
            if scene_part2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > scene_part2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    scene_part2.tStop = t  # not accounting for scr refresh
                    scene_part2.tStopRefresh = tThisFlipGlobal  # on global time
                    scene_part2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'scene_part2.stopped')
                    # update status
                    scene_part2.status = FINISHED
                    scene_part2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p2" ---
        for thisComponent in p2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p2
        p2.tStop = globalClock.getTime(format='float')
        p2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p2.stopped', p2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if p2.maxDurationReached:
            routineTimer.addTime(-p2.maxDuration)
        elif p2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "part2_question" ---
        # create an object to store info about Routine part2_question
        part2_question = data.Routine(
            name='part2_question',
            components=[text_part2_question, key_part2],
        )
        part2_question.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_part2_txt
        text_part2_qst = (
            "What is the name of the scene?\n\n"
            f"1. {scene_JA_lbl}\n"
            f"2. {scene_JB_lbl}\n"
            f"3. {scene_UA_lbl}\n"
            f"4. {scene_UB_lbl}"
        )
        text_part2_question.setText(text_part2_qst)
        # create starting attributes for key_part2
        key_part2.keys = []
        key_part2.rt = []
        _key_part2_allKeys = []
        # store start times for part2_question
        part2_question.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        part2_question.tStart = globalClock.getTime(format='float')
        part2_question.status = STARTED
        thisExp.addData('part2_question.started', part2_question.tStart)
        part2_question.maxDuration = None
        # keep track of which components have finished
        part2_questionComponents = part2_question.components
        for thisComponent in part2_question.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "part2_question" ---
        thisExp.currentRoutine = part2_question
        part2_question.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_part2_question* updates
            
            # if text_part2_question is starting this frame...
            if text_part2_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_part2_question.frameNStart = frameN  # exact frame index
                text_part2_question.tStart = t  # local t and not account for scr refresh
                text_part2_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_part2_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_part2_question.started')
                # update status
                text_part2_question.status = STARTED
                text_part2_question.setAutoDraw(True)
            
            # if text_part2_question is active this frame...
            if text_part2_question.status == STARTED:
                # update params
                pass
            
            # *key_part2* updates
            waitOnFlip = False
            
            # if key_part2 is starting this frame...
            if key_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_part2.frameNStart = frameN  # exact frame index
                key_part2.tStart = t  # local t and not account for scr refresh
                key_part2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_part2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_part2.started')
                # update status
                key_part2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_part2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_part2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_part2.status == STARTED and not waitOnFlip:
                theseKeys = key_part2.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_part2_allKeys.extend(theseKeys)
                if len(_key_part2_allKeys):
                    key_part2.keys = _key_part2_allKeys[-1].name  # just the last key pressed
                    key_part2.rt = _key_part2_allKeys[-1].rt
                    key_part2.duration = _key_part2_allKeys[-1].duration
                    # was this correct?
                    if (key_part2.keys == str(correct_key)) or (key_part2.keys == correct_key):
                        key_part2.corr = 1
                    else:
                        key_part2.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=part2_question,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                part2_question.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if part2_question.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in part2_question.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "part2_question" ---
        for thisComponent in part2_question.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for part2_question
        part2_question.tStop = globalClock.getTime(format='float')
        part2_question.tStopRefresh = tThisFlipGlobal
        thisExp.addData('part2_question.stopped', part2_question.tStop)
        # check responses
        if key_part2.keys in ['', [], None]:  # No response was made
            key_part2.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_part2.corr = 1;  # correct non-response
            else:
               key_part2.corr = 0;  # failed to respond (incorrectly)
        # store data for part2_loop (TrialHandler)
        part2_loop.addData('key_part2.keys',key_part2.keys)
        part2_loop.addData('key_part2.corr', key_part2.corr)
        if key_part2.keys != None:  # we had a response
            part2_loop.addData('key_part2.rt', key_part2.rt)
            part2_loop.addData('key_part2.duration', key_part2.duration)
        # Run 'End Routine' code from code_part2
        logic_vec[int(correct_key)-1] = key_part2.corr
        
        if all(logic_vec):
            part2_loop.finished = True
        # the Routine "part2_question" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "part2_feedback" ---
        # create an object to store info about Routine part2_feedback
        part2_feedback = data.Routine(
            name='part2_feedback',
            components=[text_part2_feedback],
        )
        part2_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_part2_feedback
        if not key_part2.keys :
            msg="Failed to respond"
        elif key_part2.corr:
            msg="Correct!"
        else:
            msg="Incorrect!"
        text_part2_feedback.setText(msg)
        # store start times for part2_feedback
        part2_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        part2_feedback.tStart = globalClock.getTime(format='float')
        part2_feedback.status = STARTED
        thisExp.addData('part2_feedback.started', part2_feedback.tStart)
        part2_feedback.maxDuration = None
        # keep track of which components have finished
        part2_feedbackComponents = part2_feedback.components
        for thisComponent in part2_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "part2_feedback" ---
        thisExp.currentRoutine = part2_feedback
        part2_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_part2_feedback* updates
            
            # if text_part2_feedback is starting this frame...
            if text_part2_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_part2_feedback.frameNStart = frameN  # exact frame index
                text_part2_feedback.tStart = t  # local t and not account for scr refresh
                text_part2_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_part2_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_part2_feedback.started')
                # update status
                text_part2_feedback.status = STARTED
                text_part2_feedback.setAutoDraw(True)
            
            # if text_part2_feedback is active this frame...
            if text_part2_feedback.status == STARTED:
                # update params
                pass
            
            # if text_part2_feedback is stopping this frame...
            if text_part2_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_part2_feedback.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_part2_feedback.tStop = t  # not accounting for scr refresh
                    text_part2_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    text_part2_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_part2_feedback.stopped')
                    # update status
                    text_part2_feedback.status = FINISHED
                    text_part2_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=part2_feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                part2_feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if part2_feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in part2_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "part2_feedback" ---
        for thisComponent in part2_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for part2_feedback
        part2_feedback.tStop = globalClock.getTime(format='float')
        part2_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('part2_feedback.stopped', part2_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if part2_feedback.maxDurationReached:
            routineTimer.addTime(-part2_feedback.maxDuration)
        elif part2_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPart2_loop as finished
        if hasattr(thisPart2_loop, 'status'):
            thisPart2_loop.status = FINISHED
        # if awaiting a pause, pause now
        if part2_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            part2_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'part2_loop'
    part2_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "part3_instructions_2" ---
    # create an object to store info about Routine part3_instructions_2
    part3_instructions_2 = data.Routine(
        name='part3_instructions_2',
        components=[text_part3_instructions_2, key_part3_instructions_2],
    )
    part3_instructions_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_part3_instructions_2
    key_part3_instructions_2.keys = []
    key_part3_instructions_2.rt = []
    _key_part3_instructions_2_allKeys = []
    # store start times for part3_instructions_2
    part3_instructions_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    part3_instructions_2.tStart = globalClock.getTime(format='float')
    part3_instructions_2.status = STARTED
    thisExp.addData('part3_instructions_2.started', part3_instructions_2.tStart)
    part3_instructions_2.maxDuration = None
    # keep track of which components have finished
    part3_instructions_2Components = part3_instructions_2.components
    for thisComponent in part3_instructions_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "part3_instructions_2" ---
    thisExp.currentRoutine = part3_instructions_2
    part3_instructions_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_part3_instructions_2* updates
        
        # if text_part3_instructions_2 is starting this frame...
        if text_part3_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_part3_instructions_2.frameNStart = frameN  # exact frame index
            text_part3_instructions_2.tStart = t  # local t and not account for scr refresh
            text_part3_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_part3_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_part3_instructions_2.started')
            # update status
            text_part3_instructions_2.status = STARTED
            text_part3_instructions_2.setAutoDraw(True)
        
        # if text_part3_instructions_2 is active this frame...
        if text_part3_instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_part3_instructions_2* updates
        waitOnFlip = False
        
        # if key_part3_instructions_2 is starting this frame...
        if key_part3_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_part3_instructions_2.frameNStart = frameN  # exact frame index
            key_part3_instructions_2.tStart = t  # local t and not account for scr refresh
            key_part3_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_part3_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_part3_instructions_2.started')
            # update status
            key_part3_instructions_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_part3_instructions_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_part3_instructions_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_part3_instructions_2.status == STARTED and not waitOnFlip:
            theseKeys = key_part3_instructions_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_part3_instructions_2_allKeys.extend(theseKeys)
            if len(_key_part3_instructions_2_allKeys):
                key_part3_instructions_2.keys = _key_part3_instructions_2_allKeys[-1].name  # just the last key pressed
                key_part3_instructions_2.rt = _key_part3_instructions_2_allKeys[-1].rt
                key_part3_instructions_2.duration = _key_part3_instructions_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=part3_instructions_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            part3_instructions_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if part3_instructions_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in part3_instructions_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "part3_instructions_2" ---
    for thisComponent in part3_instructions_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for part3_instructions_2
    part3_instructions_2.tStop = globalClock.getTime(format='float')
    part3_instructions_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('part3_instructions_2.stopped', part3_instructions_2.tStop)
    # check responses
    if key_part3_instructions_2.keys in ['', [], None]:  # No response was made
        key_part3_instructions_2.keys = None
    thisExp.addData('key_part3_instructions_2.keys',key_part3_instructions_2.keys)
    if key_part3_instructions_2.keys != None:  # we had a response
        thisExp.addData('key_part3_instructions_2.rt', key_part3_instructions_2.rt)
        thisExp.addData('key_part3_instructions_2.duration', key_part3_instructions_2.duration)
    thisExp.nextEntry()
    # the Routine "part3_instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    p3_scene_loop = data.TrialHandler2(
        name='p3_scene_loop',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(p3_scene_loop)  # add the loop to the experiment
    thisP3_scene_loop = p3_scene_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisP3_scene_loop.rgb)
    if thisP3_scene_loop != None:
        for paramName in thisP3_scene_loop:
            globals()[paramName] = thisP3_scene_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisP3_scene_loop in p3_scene_loop:
        p3_scene_loop.status = STARTED
        if hasattr(thisP3_scene_loop, 'status'):
            thisP3_scene_loop.status = STARTED
        currentLoop = p3_scene_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisP3_scene_loop.rgb)
        if thisP3_scene_loop != None:
            for paramName in thisP3_scene_loop:
                globals()[paramName] = thisP3_scene_loop[paramName]
        
        # --- Prepare to start Routine "p3_reset_vis_vars" ---
        # create an object to store info about Routine p3_reset_vis_vars
        p3_reset_vis_vars = data.Routine(
            name='p3_reset_vis_vars',
            components=[],
        )
        p3_reset_vis_vars.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        # scene order
        labelStr = ["JA", "JB", "UA", "UB"]
        
        # randomize order only once
        if count == 0:
            ordered_labels = labelStr.copy()
            random.shuffle(ordered_labels)
            print("Visualization order:", ordered_labels)
        
        # get current scene
        lbl = ordered_labels[count]
        
        # load correct participant visualization file
        vis_table = f"stimuli/vis_lists/visualization_{lbl}_sub_{expInfo['participant']}.csv"
        
        # get correct scene name from condition list
        if lbl == "JA":
            label_current = scene_JA_lbl
        elif lbl == "JB":
            label_current = scene_JB_lbl
        elif lbl == "UA":
            label_current = scene_UA_lbl
        elif lbl == "UB":
            label_current = scene_UB_lbl
        
        # bookkeeping
        logic_vec_ans = [False] * 4
        
        # advance scene counter
        count += 1
        
        # mark last scene
        last_scene = 0 if count >= 4 else 1
        # store start times for p3_reset_vis_vars
        p3_reset_vis_vars.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p3_reset_vis_vars.tStart = globalClock.getTime(format='float')
        p3_reset_vis_vars.status = STARTED
        thisExp.addData('p3_reset_vis_vars.started', p3_reset_vis_vars.tStart)
        p3_reset_vis_vars.maxDuration = None
        # keep track of which components have finished
        p3_reset_vis_varsComponents = p3_reset_vis_vars.components
        for thisComponent in p3_reset_vis_vars.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p3_reset_vis_vars" ---
        thisExp.currentRoutine = p3_reset_vis_vars
        p3_reset_vis_vars.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisP3_scene_loop, 'status') and thisP3_scene_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p3_reset_vis_vars,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p3_reset_vis_vars.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p3_reset_vis_vars.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p3_reset_vis_vars.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p3_reset_vis_vars" ---
        for thisComponent in p3_reset_vis_vars.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p3_reset_vis_vars
        p3_reset_vis_vars.tStop = globalClock.getTime(format='float')
        p3_reset_vis_vars.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p3_reset_vis_vars.stopped', p3_reset_vis_vars.tStop)
        # the Routine "p3_reset_vis_vars" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        p3_vis_loop = data.TrialHandler2(
            name='p3_vis_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(vis_table), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(p3_vis_loop)  # add the loop to the experiment
        thisP3_vis_loop = p3_vis_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisP3_vis_loop.rgb)
        if thisP3_vis_loop != None:
            for paramName in thisP3_vis_loop:
                globals()[paramName] = thisP3_vis_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisP3_vis_loop in p3_vis_loop:
            p3_vis_loop.status = STARTED
            if hasattr(thisP3_vis_loop, 'status'):
                thisP3_vis_loop.status = STARTED
            currentLoop = p3_vis_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisP3_vis_loop.rgb)
            if thisP3_vis_loop != None:
                for paramName in thisP3_vis_loop:
                    globals()[paramName] = thisP3_vis_loop[paramName]
            
            # --- Prepare to start Routine "p3_skip_answered_questions" ---
            # create an object to store info about Routine p3_skip_answered_questions
            p3_skip_answered_questions = data.Routine(
                name='p3_skip_answered_questions',
                components=[],
            )
            p3_skip_answered_questions.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_6
            if qst_num is not None and qst_num != '':
                idx = int(qst_num) - 1  
            
                if 0 <= idx < len(logic_vec_ans):
                    if logic_vec_ans[idx]:
                        skipTrial = True
                        print('should skip question', qst_num)
                        print("qst_num:", qst_num, "skip:", logic_vec_ans[idx])
                    else:
                        skipTrial = False
                else:
                    skipTrial = False
            else:
                skipTrial = False
                
                # force PsychoPy to load image into GPU BEFORE drawing
            _ = visual.ImageStim(win, image=scene_left)
            _ = visual.ImageStim(win, image=scene_right)
            # store start times for p3_skip_answered_questions
            p3_skip_answered_questions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_skip_answered_questions.tStart = globalClock.getTime(format='float')
            p3_skip_answered_questions.status = STARTED
            thisExp.addData('p3_skip_answered_questions.started', p3_skip_answered_questions.tStart)
            p3_skip_answered_questions.maxDuration = None
            # keep track of which components have finished
            p3_skip_answered_questionsComponents = p3_skip_answered_questions.components
            for thisComponent in p3_skip_answered_questions.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_skip_answered_questions" ---
            thisExp.currentRoutine = p3_skip_answered_questions
            p3_skip_answered_questions.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisP3_vis_loop, 'status') and thisP3_vis_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_skip_answered_questions,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_skip_answered_questions.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_skip_answered_questions.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_skip_answered_questions.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_skip_answered_questions" ---
            for thisComponent in p3_skip_answered_questions.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_skip_answered_questions
            p3_skip_answered_questions.tStop = globalClock.getTime(format='float')
            p3_skip_answered_questions.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_skip_answered_questions.stopped', p3_skip_answered_questions.tStop)
            # the Routine "p3_skip_answered_questions" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "p3_blank1000" ---
            # create an object to store info about Routine p3_blank1000
            p3_blank1000 = data.Routine(
                name='p3_blank1000',
                components=[text],
            )
            p3_blank1000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_6
            
            if skipTrial:
                continueRoutine = False
            # store start times for p3_blank1000
            p3_blank1000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_blank1000.tStart = globalClock.getTime(format='float')
            p3_blank1000.status = STARTED
            thisExp.addData('p3_blank1000.started', p3_blank1000.tStart)
            p3_blank1000.maxDuration = None
            # keep track of which components have finished
            p3_blank1000Components = p3_blank1000.components
            for thisComponent in p3_blank1000.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_blank1000" ---
            thisExp.currentRoutine = p3_blank1000
            p3_blank1000.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisP3_vis_loop, 'status') and thisP3_vis_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_blank1000,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_blank1000.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_blank1000.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_blank1000.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_blank1000" ---
            for thisComponent in p3_blank1000.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_blank1000
            p3_blank1000.tStop = globalClock.getTime(format='float')
            p3_blank1000.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_blank1000.stopped', p3_blank1000.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if p3_blank1000.maxDurationReached:
                routineTimer.addTime(-p3_blank1000.maxDuration)
            elif p3_blank1000.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "p3_visualize" ---
            # create an object to store info about Routine p3_visualize
            p3_visualize = data.Routine(
                name='p3_visualize',
                components=[text_p3_visualize, textbox_scene_name_2],
            )
            p3_visualize.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from p3_scene_assign
            if skipTrial:
                continueRoutine = False
            
            
            
            
            textbox_scene_name_2.reset()
            textbox_scene_name_2.setText(label_current)
            # store start times for p3_visualize
            p3_visualize.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_visualize.tStart = globalClock.getTime(format='float')
            p3_visualize.status = STARTED
            thisExp.addData('p3_visualize.started', p3_visualize.tStart)
            p3_visualize.maxDuration = None
            # keep track of which components have finished
            p3_visualizeComponents = p3_visualize.components
            for thisComponent in p3_visualize.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_visualize" ---
            thisExp.currentRoutine = p3_visualize
            p3_visualize.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # if trial has changed, end Routine now
                if hasattr(thisP3_vis_loop, 'status') and thisP3_vis_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_p3_visualize* updates
                
                # if text_p3_visualize is starting this frame...
                if text_p3_visualize.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_p3_visualize.frameNStart = frameN  # exact frame index
                    text_p3_visualize.tStart = t  # local t and not account for scr refresh
                    text_p3_visualize.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_p3_visualize, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_p3_visualize.started')
                    # update status
                    text_p3_visualize.status = STARTED
                    text_p3_visualize.setAutoDraw(True)
                
                # if text_p3_visualize is active this frame...
                if text_p3_visualize.status == STARTED:
                    # update params
                    pass
                
                # if text_p3_visualize is stopping this frame...
                if text_p3_visualize.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_p3_visualize.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_p3_visualize.tStop = t  # not accounting for scr refresh
                        text_p3_visualize.tStopRefresh = tThisFlipGlobal  # on global time
                        text_p3_visualize.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_p3_visualize.stopped')
                        # update status
                        text_p3_visualize.status = FINISHED
                        text_p3_visualize.setAutoDraw(False)
                
                # *textbox_scene_name_2* updates
                
                # if textbox_scene_name_2 is starting this frame...
                if textbox_scene_name_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_scene_name_2.frameNStart = frameN  # exact frame index
                    textbox_scene_name_2.tStart = t  # local t and not account for scr refresh
                    textbox_scene_name_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_scene_name_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_scene_name_2.started')
                    # update status
                    textbox_scene_name_2.status = STARTED
                    textbox_scene_name_2.setAutoDraw(True)
                
                # if textbox_scene_name_2 is active this frame...
                if textbox_scene_name_2.status == STARTED:
                    # update params
                    pass
                
                # if textbox_scene_name_2 is stopping this frame...
                if textbox_scene_name_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textbox_scene_name_2.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        textbox_scene_name_2.tStop = t  # not accounting for scr refresh
                        textbox_scene_name_2.tStopRefresh = tThisFlipGlobal  # on global time
                        textbox_scene_name_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textbox_scene_name_2.stopped')
                        # update status
                        textbox_scene_name_2.status = FINISHED
                        textbox_scene_name_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_visualize,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_visualize.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_visualize.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_visualize.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_visualize" ---
            for thisComponent in p3_visualize.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_visualize
            p3_visualize.tStop = globalClock.getTime(format='float')
            p3_visualize.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_visualize.stopped', p3_visualize.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if p3_visualize.maxDurationReached:
                routineTimer.addTime(-p3_visualize.maxDuration)
            elif p3_visualize.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "p3_rate" ---
            # create an object to store info about Routine p3_rate
            p3_rate = data.Routine(
                name='p3_rate',
                components=[text_rate_2, key_rate_2],
            )
            p3_rate.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_7
            
            
            if skipTrial:
                continueRoutine = False
            # create starting attributes for key_rate_2
            key_rate_2.keys = []
            key_rate_2.rt = []
            _key_rate_2_allKeys = []
            # store start times for p3_rate
            p3_rate.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_rate.tStart = globalClock.getTime(format='float')
            p3_rate.status = STARTED
            thisExp.addData('p3_rate.started', p3_rate.tStart)
            p3_rate.maxDuration = None
            # keep track of which components have finished
            p3_rateComponents = p3_rate.components
            for thisComponent in p3_rate.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_rate" ---
            thisExp.currentRoutine = p3_rate
            p3_rate.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisP3_vis_loop, 'status') and thisP3_vis_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_rate_2* updates
                
                # if text_rate_2 is starting this frame...
                if text_rate_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_rate_2.frameNStart = frameN  # exact frame index
                    text_rate_2.tStart = t  # local t and not account for scr refresh
                    text_rate_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_rate_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_rate_2.started')
                    # update status
                    text_rate_2.status = STARTED
                    text_rate_2.setAutoDraw(True)
                
                # if text_rate_2 is active this frame...
                if text_rate_2.status == STARTED:
                    # update params
                    pass
                
                # *key_rate_2* updates
                waitOnFlip = False
                
                # if key_rate_2 is starting this frame...
                if key_rate_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_rate_2.frameNStart = frameN  # exact frame index
                    key_rate_2.tStart = t  # local t and not account for scr refresh
                    key_rate_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_rate_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_rate_2.started')
                    # update status
                    key_rate_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_rate_2.clock.reset)  # t=0 on next screen flip
                if key_rate_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_rate_2.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                    _key_rate_2_allKeys.extend(theseKeys)
                    if len(_key_rate_2_allKeys):
                        key_rate_2.keys = _key_rate_2_allKeys[-1].name  # just the last key pressed
                        key_rate_2.rt = _key_rate_2_allKeys[-1].rt
                        key_rate_2.duration = _key_rate_2_allKeys[-1].duration
                        # was this correct?
                        if (key_rate_2.keys == str("'3', '4'")) or (key_rate_2.keys == "'3', '4'"):
                            key_rate_2.corr = 1
                        else:
                            key_rate_2.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_rate,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_rate.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_rate.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_rate.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_rate" ---
            for thisComponent in p3_rate.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_rate
            p3_rate.tStop = globalClock.getTime(format='float')
            p3_rate.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_rate.stopped', p3_rate.tStop)
            # check responses
            if key_rate_2.keys in ['', [], None]:  # No response was made
                key_rate_2.keys = None
                # was no response the correct answer?!
                if str("'3', '4'").lower() == 'none':
                   key_rate_2.corr = 1;  # correct non-response
                else:
                   key_rate_2.corr = 0;  # failed to respond (incorrectly)
            # store data for p3_vis_loop (TrialHandler)
            p3_vis_loop.addData('key_rate_2.keys',key_rate_2.keys)
            p3_vis_loop.addData('key_rate_2.corr', key_rate_2.corr)
            if key_rate_2.keys != None:  # we had a response
                p3_vis_loop.addData('key_rate_2.rt', key_rate_2.rt)
                p3_vis_loop.addData('key_rate_2.duration', key_rate_2.duration)
            # the Routine "p3_rate" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "p3_question" ---
            # create an object to store info about Routine p3_question
            p3_question = data.Routine(
                name='p3_question',
                components=[text_question_2, text_option_2, key_question_2, text_5],
            )
            p3_question.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip
            key_question_2.keys = None
            key_question_2.rt = None
            
            if skipTrial:
                continueRoutine = False
                
              
            text_question_2.setText(question)
            text_option_2.setText(options)
            # create starting attributes for key_question_2
            key_question_2.keys = []
            key_question_2.rt = []
            _key_question_2_allKeys = []
            # Run 'Begin Routine' code from code_part3
            rate_ans = None
            myReps = 1
            # store start times for p3_question
            p3_question.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_question.tStart = globalClock.getTime(format='float')
            p3_question.status = STARTED
            thisExp.addData('p3_question.started', p3_question.tStart)
            p3_question.maxDuration = None
            # keep track of which components have finished
            p3_questionComponents = p3_question.components
            for thisComponent in p3_question.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_question" ---
            thisExp.currentRoutine = p3_question
            p3_question.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisP3_vis_loop, 'status') and thisP3_vis_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_question_2* updates
                
                # if text_question_2 is starting this frame...
                if text_question_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_question_2.frameNStart = frameN  # exact frame index
                    text_question_2.tStart = t  # local t and not account for scr refresh
                    text_question_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_question_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_question_2.started')
                    # update status
                    text_question_2.status = STARTED
                    text_question_2.setAutoDraw(True)
                
                # if text_question_2 is active this frame...
                if text_question_2.status == STARTED:
                    # update params
                    pass
                
                # *text_option_2* updates
                
                # if text_option_2 is starting this frame...
                if text_option_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_option_2.frameNStart = frameN  # exact frame index
                    text_option_2.tStart = t  # local t and not account for scr refresh
                    text_option_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_option_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_option_2.started')
                    # update status
                    text_option_2.status = STARTED
                    text_option_2.setAutoDraw(True)
                
                # if text_option_2 is active this frame...
                if text_option_2.status == STARTED:
                    # update params
                    pass
                
                # *key_question_2* updates
                
                # if key_question_2 is starting this frame...
                if key_question_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_question_2.frameNStart = frameN  # exact frame index
                    key_question_2.tStart = t  # local t and not account for scr refresh
                    key_question_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_question_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('key_question_2.started', t)
                    # update status
                    key_question_2.status = STARTED
                    # keyboard checking is just starting
                    key_question_2.clock.reset()  # now t=0
                    key_question_2.clearEvents(eventType='keyboard')
                if key_question_2.status == STARTED:
                    theseKeys = key_question_2.getKeys(keyList=['1','2', '3', '4'], ignoreKeys=["escape"], waitRelease=False)
                    _key_question_2_allKeys.extend(theseKeys)
                    if len(_key_question_2_allKeys):
                        key_question_2.keys = _key_question_2_allKeys[-1].name  # just the last key pressed
                        key_question_2.rt = _key_question_2_allKeys[-1].rt
                        key_question_2.duration = _key_question_2_allKeys[-1].duration
                        # was this correct?
                        if (key_question_2.keys == str(correct_answer)) or (key_question_2.keys == correct_answer):
                            key_question_2.corr = 1
                        else:
                            key_question_2.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_5* updates
                
                # if text_5 is starting this frame...
                if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_5.frameNStart = frameN  # exact frame index
                    text_5.tStart = t  # local t and not account for scr refresh
                    text_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_5.started')
                    # update status
                    text_5.status = STARTED
                    text_5.setAutoDraw(True)
                
                # if text_5 is active this frame...
                if text_5.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_question,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_question.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_question.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_question.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_question" ---
            for thisComponent in p3_question.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_question
            p3_question.tStop = globalClock.getTime(format='float')
            p3_question.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_question.stopped', p3_question.tStop)
            # check responses
            if key_question_2.keys in ['', [], None]:  # No response was made
                key_question_2.keys = None
                # was no response the correct answer?!
                if str(correct_answer).lower() == 'none':
                   key_question_2.corr = 1;  # correct non-response
                else:
                   key_question_2.corr = 0;  # failed to respond (incorrectly)
            # store data for p3_vis_loop (TrialHandler)
            p3_vis_loop.addData('key_question_2.keys',key_question_2.keys)
            p3_vis_loop.addData('key_question_2.corr', key_question_2.corr)
            if key_question_2.keys != None:  # we had a response
                p3_vis_loop.addData('key_question_2.rt', key_question_2.rt)
                p3_vis_loop.addData('key_question_2.duration', key_question_2.duration)
            # Run 'End Routine' code from code_part3
            rate_ans = key_rate_2.keys
            
            if key_question_2.corr == 1:
                logic_vec_ans[int(qst_num) - 1] = True
                
                if rate_ans == "3" or rate_ans == "4":
                    myReps = 0
                else:
                    myReps = 1
            else:
                myReps = 1
            
            # Only use .finished to END loop, not control flow mid-loop
            if all(logic_vec_ans) and (rate_ans == "3" or rate_ans == "4"):
                p3_vis_loop.finished = True
            # the Routine "p3_question" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            skip_loop = data.TrialHandler2(
                name='skip_loop',
                nReps=myReps, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(skip_loop)  # add the loop to the experiment
            thisSkip_loop = skip_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSkip_loop.rgb)
            if thisSkip_loop != None:
                for paramName in thisSkip_loop:
                    globals()[paramName] = thisSkip_loop[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSkip_loop in skip_loop:
                skip_loop.status = STARTED
                if hasattr(thisSkip_loop, 'status'):
                    thisSkip_loop.status = STARTED
                currentLoop = skip_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSkip_loop.rgb)
                if thisSkip_loop != None:
                    for paramName in thisSkip_loop:
                        globals()[paramName] = thisSkip_loop[paramName]
                
                # --- Prepare to start Routine "p3_study_again" ---
                # create an object to store info about Routine p3_study_again
                p3_study_again = data.Routine(
                    name='p3_study_again',
                    components=[study_scene_left, study_scene_right, study_name1_2, key_sceneLearn, text_6, text_7],
                )
                p3_study_again.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from code_8
                print(scene_left, scene_right)
                
                if 'skipTrial' in globals() and skipTrial:
                    continueRoutine = False
                study_scene_left.setImage(scene_left)
                study_scene_right.setImage(scene_right)
                study_name1_2.setPos((0, .35))
                study_name1_2.setText(label_current)
                # create starting attributes for key_sceneLearn
                key_sceneLearn.keys = []
                key_sceneLearn.rt = []
                _key_sceneLearn_allKeys = []
                # store start times for p3_study_again
                p3_study_again.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                p3_study_again.tStart = globalClock.getTime(format='float')
                p3_study_again.status = STARTED
                thisExp.addData('p3_study_again.started', p3_study_again.tStart)
                p3_study_again.maxDuration = None
                # keep track of which components have finished
                p3_study_againComponents = p3_study_again.components
                for thisComponent in p3_study_again.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "p3_study_again" ---
                thisExp.currentRoutine = p3_study_again
                p3_study_again.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisSkip_loop, 'status') and thisSkip_loop.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *study_scene_left* updates
                    
                    # if study_scene_left is starting this frame...
                    if study_scene_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        study_scene_left.frameNStart = frameN  # exact frame index
                        study_scene_left.tStart = t  # local t and not account for scr refresh
                        study_scene_left.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(study_scene_left, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'study_scene_left.started')
                        # update status
                        study_scene_left.status = STARTED
                        study_scene_left.setAutoDraw(True)
                    
                    # if study_scene_left is active this frame...
                    if study_scene_left.status == STARTED:
                        # update params
                        pass
                    
                    # *study_scene_right* updates
                    
                    # if study_scene_right is starting this frame...
                    if study_scene_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        study_scene_right.frameNStart = frameN  # exact frame index
                        study_scene_right.tStart = t  # local t and not account for scr refresh
                        study_scene_right.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(study_scene_right, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'study_scene_right.started')
                        # update status
                        study_scene_right.status = STARTED
                        study_scene_right.setAutoDraw(True)
                    
                    # if study_scene_right is active this frame...
                    if study_scene_right.status == STARTED:
                        # update params
                        pass
                    
                    # *study_name1_2* updates
                    
                    # if study_name1_2 is starting this frame...
                    if study_name1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        study_name1_2.frameNStart = frameN  # exact frame index
                        study_name1_2.tStart = t  # local t and not account for scr refresh
                        study_name1_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(study_name1_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'study_name1_2.started')
                        # update status
                        study_name1_2.status = STARTED
                        study_name1_2.setAutoDraw(True)
                    
                    # if study_name1_2 is active this frame...
                    if study_name1_2.status == STARTED:
                        # update params
                        pass
                    
                    # *key_sceneLearn* updates
                    waitOnFlip = False
                    
                    # if key_sceneLearn is starting this frame...
                    if key_sceneLearn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_sceneLearn.frameNStart = frameN  # exact frame index
                        key_sceneLearn.tStart = t  # local t and not account for scr refresh
                        key_sceneLearn.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_sceneLearn, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_sceneLearn.started')
                        # update status
                        key_sceneLearn.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_sceneLearn.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_sceneLearn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if key_sceneLearn.status == STARTED and not waitOnFlip:
                        theseKeys = key_sceneLearn.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _key_sceneLearn_allKeys.extend(theseKeys)
                        if len(_key_sceneLearn_allKeys):
                            key_sceneLearn.keys = _key_sceneLearn_allKeys[-1].name  # just the last key pressed
                            key_sceneLearn.rt = _key_sceneLearn_allKeys[-1].rt
                            key_sceneLearn.duration = _key_sceneLearn_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *text_6* updates
                    
                    # if text_6 is starting this frame...
                    if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_6.frameNStart = frameN  # exact frame index
                        text_6.tStart = t  # local t and not account for scr refresh
                        text_6.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_6.started')
                        # update status
                        text_6.status = STARTED
                        text_6.setAutoDraw(True)
                    
                    # if text_6 is active this frame...
                    if text_6.status == STARTED:
                        # update params
                        pass
                    
                    # *text_7* updates
                    
                    # if text_7 is starting this frame...
                    if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_7.frameNStart = frameN  # exact frame index
                        text_7.tStart = t  # local t and not account for scr refresh
                        text_7.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_7.started')
                        # update status
                        text_7.status = STARTED
                        text_7.setAutoDraw(True)
                    
                    # if text_7 is active this frame...
                    if text_7.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=p3_study_again,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        p3_study_again.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if p3_study_again.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in p3_study_again.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "p3_study_again" ---
                for thisComponent in p3_study_again.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for p3_study_again
                p3_study_again.tStop = globalClock.getTime(format='float')
                p3_study_again.tStopRefresh = tThisFlipGlobal
                thisExp.addData('p3_study_again.stopped', p3_study_again.tStop)
                # check responses
                if key_sceneLearn.keys in ['', [], None]:  # No response was made
                    key_sceneLearn.keys = None
                skip_loop.addData('key_sceneLearn.keys',key_sceneLearn.keys)
                if key_sceneLearn.keys != None:  # we had a response
                    skip_loop.addData('key_sceneLearn.rt', key_sceneLearn.rt)
                    skip_loop.addData('key_sceneLearn.duration', key_sceneLearn.duration)
                # the Routine "p3_study_again" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisSkip_loop as finished
                if hasattr(thisSkip_loop, 'status'):
                    thisSkip_loop.status = FINISHED
                # if awaiting a pause, pause now
                if skip_loop.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    skip_loop.status = STARTED
                thisExp.nextEntry()
                
            # completed myReps repeats of 'skip_loop'
            skip_loop.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # mark thisP3_vis_loop as finished
            if hasattr(thisP3_vis_loop, 'status'):
                thisP3_vis_loop.status = FINISHED
            # if awaiting a pause, pause now
            if p3_vis_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                p3_vis_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'p3_vis_loop'
        p3_vis_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_msg = data.TrialHandler2(
            name='loop_msg',
            nReps=last_scene, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(loop_msg)  # add the loop to the experiment
        thisLoop_msg = loop_msg.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_msg.rgb)
        if thisLoop_msg != None:
            for paramName in thisLoop_msg:
                globals()[paramName] = thisLoop_msg[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_msg in loop_msg:
            loop_msg.status = STARTED
            if hasattr(thisLoop_msg, 'status'):
                thisLoop_msg.status = STARTED
            currentLoop = loop_msg
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_msg.rgb)
            if thisLoop_msg != None:
                for paramName in thisLoop_msg:
                    globals()[paramName] = thisLoop_msg[paramName]
            
            # --- Prepare to start Routine "p3_correct_msg" ---
            # create an object to store info about Routine p3_correct_msg
            p3_correct_msg = data.Routine(
                name='p3_correct_msg',
                components=[text_next_scene_2],
            )
            p3_correct_msg.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for p3_correct_msg
            p3_correct_msg.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p3_correct_msg.tStart = globalClock.getTime(format='float')
            p3_correct_msg.status = STARTED
            thisExp.addData('p3_correct_msg.started', p3_correct_msg.tStart)
            p3_correct_msg.maxDuration = None
            # keep track of which components have finished
            p3_correct_msgComponents = p3_correct_msg.components
            for thisComponent in p3_correct_msg.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "p3_correct_msg" ---
            thisExp.currentRoutine = p3_correct_msg
            p3_correct_msg.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # if trial has changed, end Routine now
                if hasattr(thisLoop_msg, 'status') and thisLoop_msg.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_next_scene_2* updates
                
                # if text_next_scene_2 is starting this frame...
                if text_next_scene_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_next_scene_2.frameNStart = frameN  # exact frame index
                    text_next_scene_2.tStart = t  # local t and not account for scr refresh
                    text_next_scene_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_next_scene_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_next_scene_2.started')
                    # update status
                    text_next_scene_2.status = STARTED
                    text_next_scene_2.setAutoDraw(True)
                
                # if text_next_scene_2 is active this frame...
                if text_next_scene_2.status == STARTED:
                    # update params
                    pass
                
                # if text_next_scene_2 is stopping this frame...
                if text_next_scene_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_next_scene_2.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_next_scene_2.tStop = t  # not accounting for scr refresh
                        text_next_scene_2.tStopRefresh = tThisFlipGlobal  # on global time
                        text_next_scene_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_next_scene_2.stopped')
                        # update status
                        text_next_scene_2.status = FINISHED
                        text_next_scene_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=p3_correct_msg,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p3_correct_msg.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p3_correct_msg.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p3_correct_msg.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p3_correct_msg" ---
            for thisComponent in p3_correct_msg.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p3_correct_msg
            p3_correct_msg.tStop = globalClock.getTime(format='float')
            p3_correct_msg.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p3_correct_msg.stopped', p3_correct_msg.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if p3_correct_msg.maxDurationReached:
                routineTimer.addTime(-p3_correct_msg.maxDuration)
            elif p3_correct_msg.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            # mark thisLoop_msg as finished
            if hasattr(thisLoop_msg, 'status'):
                thisLoop_msg.status = FINISHED
            # if awaiting a pause, pause now
            if loop_msg.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                loop_msg.status = STARTED
            thisExp.nextEntry()
            
        # completed last_scene repeats of 'loop_msg'
        loop_msg.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisP3_scene_loop as finished
        if hasattr(thisP3_scene_loop, 'status'):
            thisP3_scene_loop.status = FINISHED
        # if awaiting a pause, pause now
        if p3_scene_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            p3_scene_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'p3_scene_loop'
    p3_scene_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p4_instructions" ---
    # create an object to store info about Routine p4_instructions
    p4_instructions = data.Routine(
        name='p4_instructions',
        components=[text_part4_instructions, key_part4_instructions],
    )
    p4_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_training_vars
    # allocate participant to relatedness response key according
    # to counterbalanced_vars.xlsx
    
    # =========================================
    # LOAD COUNTERBALANCING FILE
    # =========================================
    
    import pandas as pd
    
    counterbalance_file = 'counterbalanced_vars_behav.csv'
    
    df = pd.read_csv(counterbalance_file, header=1)
    
    # clean column names
    df.columns = df.columns.str.strip()
    
    # force numeric
    df['enc_relatedness_keys'] = pd.to_numeric(
        df['enc_relatedness_keys'],
        errors='coerce'
    )
    
    subnum = int(expInfo['participant'])
    
    # find participant row
    ind = df[df['Participant'] == subnum].index[0]
    
    print("enc_relatedness_keys:", df.loc[ind, 'enc_relatedness_keys'])
    
    # =========================================
    # ASSIGN RESPONSE KEYS
    # =========================================
    
    if df.loc[ind, 'enc_relatedness_keys'] == 0:
    
        key_related = 'q'
        key_unrelated = 'p'
    
        instruction_text = (
            'If they are related, press Q\n'
            'If they are unrelated, press P\n\n'
            'Press SPACEBAR to continue.'
        )
    
    elif df.loc[ind, 'enc_relatedness_keys'] == 1:
    
        key_related = 'p'
        key_unrelated = 'q'
    
        instruction_text = (
            'If they are related, press P\n'
            'If they are unrelated, press Q\n\n'
            'Press SPACEBAR to continue.'
        )
    
    # =========================================
    # RESPONSE KEY LIST
    # =========================================
    
    response_keys = [key_related, key_unrelated]
    # create starting attributes for key_part4_instructions
    key_part4_instructions.keys = []
    key_part4_instructions.rt = []
    _key_part4_instructions_allKeys = []
    # store start times for p4_instructions
    p4_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p4_instructions.tStart = globalClock.getTime(format='float')
    p4_instructions.status = STARTED
    thisExp.addData('p4_instructions.started', p4_instructions.tStart)
    p4_instructions.maxDuration = None
    # keep track of which components have finished
    p4_instructionsComponents = p4_instructions.components
    for thisComponent in p4_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "p4_instructions" ---
    thisExp.currentRoutine = p4_instructions
    p4_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_part4_instructions* updates
        
        # if text_part4_instructions is starting this frame...
        if text_part4_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_part4_instructions.frameNStart = frameN  # exact frame index
            text_part4_instructions.tStart = t  # local t and not account for scr refresh
            text_part4_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_part4_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_part4_instructions.started')
            # update status
            text_part4_instructions.status = STARTED
            text_part4_instructions.setAutoDraw(True)
        
        # if text_part4_instructions is active this frame...
        if text_part4_instructions.status == STARTED:
            # update params
            pass
        
        # *key_part4_instructions* updates
        waitOnFlip = False
        
        # if key_part4_instructions is starting this frame...
        if key_part4_instructions.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_part4_instructions.frameNStart = frameN  # exact frame index
            key_part4_instructions.tStart = t  # local t and not account for scr refresh
            key_part4_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_part4_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_part4_instructions.started')
            # update status
            key_part4_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_part4_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_part4_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_part4_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_part4_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_part4_instructions_allKeys.extend(theseKeys)
            if len(_key_part4_instructions_allKeys):
                key_part4_instructions.keys = _key_part4_instructions_allKeys[-1].name  # just the last key pressed
                key_part4_instructions.rt = _key_part4_instructions_allKeys[-1].rt
                key_part4_instructions.duration = _key_part4_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=p4_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p4_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p4_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p4_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p4_instructions" ---
    for thisComponent in p4_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p4_instructions
    p4_instructions.tStop = globalClock.getTime(format='float')
    p4_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p4_instructions.stopped', p4_instructions.tStop)
    # check responses
    if key_part4_instructions.keys in ['', [], None]:  # No response was made
        key_part4_instructions.keys = None
    thisExp.addData('key_part4_instructions.keys',key_part4_instructions.keys)
    if key_part4_instructions.keys != None:  # we had a response
        thisExp.addData('key_part4_instructions.rt', key_part4_instructions.rt)
        thisExp.addData('key_part4_instructions.duration', key_part4_instructions.duration)
    thisExp.nextEntry()
    # the Routine "p4_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "enc_instruction_screen" ---
    # create an object to store info about Routine enc_instruction_screen
    enc_instruction_screen = data.Routine(
        name='enc_instruction_screen',
        components=[text_enc_instruction, key_proceed],
    )
    enc_instruction_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_proceed
    key_proceed.keys = []
    key_proceed.rt = []
    _key_proceed_allKeys = []
    # store start times for enc_instruction_screen
    enc_instruction_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    enc_instruction_screen.tStart = globalClock.getTime(format='float')
    enc_instruction_screen.status = STARTED
    thisExp.addData('enc_instruction_screen.started', enc_instruction_screen.tStart)
    enc_instruction_screen.maxDuration = None
    # keep track of which components have finished
    enc_instruction_screenComponents = enc_instruction_screen.components
    for thisComponent in enc_instruction_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "enc_instruction_screen" ---
    thisExp.currentRoutine = enc_instruction_screen
    enc_instruction_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_enc_instruction* updates
        
        # if text_enc_instruction is starting this frame...
        if text_enc_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_enc_instruction.frameNStart = frameN  # exact frame index
            text_enc_instruction.tStart = t  # local t and not account for scr refresh
            text_enc_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_enc_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_enc_instruction.started')
            # update status
            text_enc_instruction.status = STARTED
            text_enc_instruction.setAutoDraw(True)
        
        # if text_enc_instruction is active this frame...
        if text_enc_instruction.status == STARTED:
            # update params
            pass
        
        # *key_proceed* updates
        waitOnFlip = False
        
        # if key_proceed is starting this frame...
        if key_proceed.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_proceed.frameNStart = frameN  # exact frame index
            key_proceed.tStart = t  # local t and not account for scr refresh
            key_proceed.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_proceed, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_proceed.started')
            # update status
            key_proceed.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_proceed.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_proceed.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_proceed.status == STARTED and not waitOnFlip:
            theseKeys = key_proceed.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_proceed_allKeys.extend(theseKeys)
            if len(_key_proceed_allKeys):
                key_proceed.keys = _key_proceed_allKeys[-1].name  # just the last key pressed
                key_proceed.rt = _key_proceed_allKeys[-1].rt
                key_proceed.duration = _key_proceed_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=enc_instruction_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            enc_instruction_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if enc_instruction_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in enc_instruction_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "enc_instruction_screen" ---
    for thisComponent in enc_instruction_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for enc_instruction_screen
    enc_instruction_screen.tStop = globalClock.getTime(format='float')
    enc_instruction_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('enc_instruction_screen.stopped', enc_instruction_screen.tStop)
    # check responses
    if key_proceed.keys in ['', [], None]:  # No response was made
        key_proceed.keys = None
    thisExp.addData('key_proceed.keys',key_proceed.keys)
    if key_proceed.keys != None:  # we had a response
        thisExp.addData('key_proceed.rt', key_proceed.rt)
        thisExp.addData('key_proceed.duration', key_proceed.duration)
    thisExp.nextEntry()
    # the Routine "enc_instruction_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "enc_inst_2" ---
    # create an object to store info about Routine enc_inst_2
    enc_inst_2 = data.Routine(
        name='enc_inst_2',
        components=[text_relatedness_keys, key_proceed_2],
    )
    enc_inst_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_relatedness_keys.setText(instruction_text)
    # create starting attributes for key_proceed_2
    key_proceed_2.keys = []
    key_proceed_2.rt = []
    _key_proceed_2_allKeys = []
    # store start times for enc_inst_2
    enc_inst_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    enc_inst_2.tStart = globalClock.getTime(format='float')
    enc_inst_2.status = STARTED
    thisExp.addData('enc_inst_2.started', enc_inst_2.tStart)
    enc_inst_2.maxDuration = None
    # keep track of which components have finished
    enc_inst_2Components = enc_inst_2.components
    for thisComponent in enc_inst_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "enc_inst_2" ---
    thisExp.currentRoutine = enc_inst_2
    enc_inst_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_relatedness_keys* updates
        
        # if text_relatedness_keys is starting this frame...
        if text_relatedness_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_relatedness_keys.frameNStart = frameN  # exact frame index
            text_relatedness_keys.tStart = t  # local t and not account for scr refresh
            text_relatedness_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_relatedness_keys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_relatedness_keys.started')
            # update status
            text_relatedness_keys.status = STARTED
            text_relatedness_keys.setAutoDraw(True)
        
        # if text_relatedness_keys is active this frame...
        if text_relatedness_keys.status == STARTED:
            # update params
            pass
        
        # *key_proceed_2* updates
        waitOnFlip = False
        
        # if key_proceed_2 is starting this frame...
        if key_proceed_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            key_proceed_2.frameNStart = frameN  # exact frame index
            key_proceed_2.tStart = t  # local t and not account for scr refresh
            key_proceed_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_proceed_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_proceed_2.started')
            # update status
            key_proceed_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_proceed_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_proceed_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_proceed_2.status == STARTED and not waitOnFlip:
            theseKeys = key_proceed_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_proceed_2_allKeys.extend(theseKeys)
            if len(_key_proceed_2_allKeys):
                key_proceed_2.keys = _key_proceed_2_allKeys[-1].name  # just the last key pressed
                key_proceed_2.rt = _key_proceed_2_allKeys[-1].rt
                key_proceed_2.duration = _key_proceed_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=enc_inst_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            enc_inst_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if enc_inst_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in enc_inst_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "enc_inst_2" ---
    for thisComponent in enc_inst_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for enc_inst_2
    enc_inst_2.tStop = globalClock.getTime(format='float')
    enc_inst_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('enc_inst_2.stopped', enc_inst_2.tStop)
    # check responses
    if key_proceed_2.keys in ['', [], None]:  # No response was made
        key_proceed_2.keys = None
    thisExp.addData('key_proceed_2.keys',key_proceed_2.keys)
    if key_proceed_2.keys != None:  # we had a response
        thisExp.addData('key_proceed_2.rt', key_proceed_2.rt)
        thisExp.addData('key_proceed_2.duration', key_proceed_2.duration)
    thisExp.nextEntry()
    # the Routine "enc_inst_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "get_stim_list" ---
    # create an object to store info about Routine get_stim_list
    get_stim_list = data.Routine(
        name='get_stim_list',
        components=[],
    )
    get_stim_list.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for get_stim_list
    get_stim_list.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    get_stim_list.tStart = globalClock.getTime(format='float')
    get_stim_list.status = STARTED
    thisExp.addData('get_stim_list.started', get_stim_list.tStart)
    get_stim_list.maxDuration = None
    # keep track of which components have finished
    get_stim_listComponents = get_stim_list.components
    for thisComponent in get_stim_list.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "get_stim_list" ---
    thisExp.currentRoutine = get_stim_list
    get_stim_list.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=get_stim_list,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            get_stim_list.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if get_stim_list.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in get_stim_list.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "get_stim_list" ---
    for thisComponent in get_stim_list.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for get_stim_list
    get_stim_list.tStop = globalClock.getTime(format='float')
    get_stim_list.tStopRefresh = tThisFlipGlobal
    thisExp.addData('get_stim_list.stopped', get_stim_list.tStop)
    thisExp.nextEntry()
    # the Routine "get_stim_list" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_enc = data.TrialHandler2(
        name='trials_enc',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(stim_list), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials_enc)  # add the loop to the experiment
    thisTrials_enc = trials_enc.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_enc.rgb)
    if thisTrials_enc != None:
        for paramName in thisTrials_enc:
            globals()[paramName] = thisTrials_enc[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_enc in trials_enc:
        trials_enc.status = STARTED
        if hasattr(thisTrials_enc, 'status'):
            thisTrials_enc.status = STARTED
        currentLoop = trials_enc
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_enc.rgb)
        if thisTrials_enc != None:
            for paramName in thisTrials_enc:
                globals()[paramName] = thisTrials_enc[paramName]
        
        # --- Prepare to start Routine "integrated_scene_presentation" ---
        # create an object to store info about Routine integrated_scene_presentation
        integrated_scene_presentation = data.Routine(
            name='integrated_scene_presentation',
            components=[image_integrated],
        )
        integrated_scene_presentation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_integrated.setImage(integrated_stimulus)
        # Run 'Begin Routine' code from code_14
        _ = visual.ImageStim(win, image=integrated_stimulus)
        # store start times for integrated_scene_presentation
        integrated_scene_presentation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        integrated_scene_presentation.tStart = globalClock.getTime(format='float')
        integrated_scene_presentation.status = STARTED
        thisExp.addData('integrated_scene_presentation.started', integrated_scene_presentation.tStart)
        integrated_scene_presentation.maxDuration = None
        # keep track of which components have finished
        integrated_scene_presentationComponents = integrated_scene_presentation.components
        for thisComponent in integrated_scene_presentation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "integrated_scene_presentation" ---
        thisExp.currentRoutine = integrated_scene_presentation
        integrated_scene_presentation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_enc, 'status') and thisTrials_enc.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_integrated* updates
            
            # if image_integrated is starting this frame...
            if image_integrated.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_integrated.frameNStart = frameN  # exact frame index
                image_integrated.tStart = t  # local t and not account for scr refresh
                image_integrated.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_integrated, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_integrated.started')
                # update status
                image_integrated.status = STARTED
                image_integrated.setAutoDraw(True)
            
            # if image_integrated is active this frame...
            if image_integrated.status == STARTED:
                # update params
                pass
            
            # if image_integrated is stopping this frame...
            if image_integrated.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_integrated.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_integrated.tStop = t  # not accounting for scr refresh
                    image_integrated.tStopRefresh = tThisFlipGlobal  # on global time
                    image_integrated.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_integrated.stopped')
                    # update status
                    image_integrated.status = FINISHED
                    image_integrated.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=integrated_scene_presentation,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                integrated_scene_presentation.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if integrated_scene_presentation.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in integrated_scene_presentation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "integrated_scene_presentation" ---
        for thisComponent in integrated_scene_presentation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for integrated_scene_presentation
        integrated_scene_presentation.tStop = globalClock.getTime(format='float')
        integrated_scene_presentation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('integrated_scene_presentation.stopped', integrated_scene_presentation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if integrated_scene_presentation.maxDurationReached:
            routineTimer.addTime(-integrated_scene_presentation.maxDuration)
        elif integrated_scene_presentation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "congruency_response" ---
        # create an object to store info about Routine congruency_response
        congruency_response = data.Routine(
            name='congruency_response',
            components=[text_congruency, key_resp_congruency],
        )
        congruency_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_congruency
        key_resp_congruency.keys = []
        key_resp_congruency.rt = []
        _key_resp_congruency_allKeys = []
        # store start times for congruency_response
        congruency_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        congruency_response.tStart = globalClock.getTime(format='float')
        congruency_response.status = STARTED
        thisExp.addData('congruency_response.started', congruency_response.tStart)
        congruency_response.maxDuration = None
        # keep track of which components have finished
        congruency_responseComponents = congruency_response.components
        for thisComponent in congruency_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "congruency_response" ---
        thisExp.currentRoutine = congruency_response
        congruency_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_enc, 'status') and thisTrials_enc.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_congruency* updates
            
            # if text_congruency is starting this frame...
            if text_congruency.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_congruency.frameNStart = frameN  # exact frame index
                text_congruency.tStart = t  # local t and not account for scr refresh
                text_congruency.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_congruency, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_congruency.started')
                # update status
                text_congruency.status = STARTED
                text_congruency.setAutoDraw(True)
            
            # if text_congruency is active this frame...
            if text_congruency.status == STARTED:
                # update params
                pass
            
            # if text_congruency is stopping this frame...
            if text_congruency.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_congruency.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_congruency.tStop = t  # not accounting for scr refresh
                    text_congruency.tStopRefresh = tThisFlipGlobal  # on global time
                    text_congruency.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_congruency.stopped')
                    # update status
                    text_congruency.status = FINISHED
                    text_congruency.setAutoDraw(False)
            
            # *key_resp_congruency* updates
            waitOnFlip = False
            
            # if key_resp_congruency is starting this frame...
            if key_resp_congruency.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_congruency.frameNStart = frameN  # exact frame index
                key_resp_congruency.tStart = t  # local t and not account for scr refresh
                key_resp_congruency.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_congruency, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_congruency.started')
                # update status
                key_resp_congruency.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_congruency.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_congruency.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_congruency is stopping this frame...
            if key_resp_congruency.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_congruency.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_congruency.tStop = t  # not accounting for scr refresh
                    key_resp_congruency.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_congruency.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_congruency.stopped')
                    # update status
                    key_resp_congruency.status = FINISHED
                    key_resp_congruency.status = FINISHED
            if key_resp_congruency.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_congruency.getKeys(keyList=['p', 'q'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_congruency_allKeys.extend(theseKeys)
                if len(_key_resp_congruency_allKeys):
                    key_resp_congruency.keys = _key_resp_congruency_allKeys[-1].name  # just the last key pressed
                    key_resp_congruency.rt = _key_resp_congruency_allKeys[-1].rt
                    key_resp_congruency.duration = _key_resp_congruency_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_congruency.keys == str(correct_key)) or (key_resp_congruency.keys == correct_key):
                        key_resp_congruency.corr = 1
                    else:
                        key_resp_congruency.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=congruency_response,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                congruency_response.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if congruency_response.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in congruency_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "congruency_response" ---
        for thisComponent in congruency_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for congruency_response
        congruency_response.tStop = globalClock.getTime(format='float')
        congruency_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('congruency_response.stopped', congruency_response.tStop)
        # check responses
        if key_resp_congruency.keys in ['', [], None]:  # No response was made
            key_resp_congruency.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_resp_congruency.corr = 1;  # correct non-response
            else:
               key_resp_congruency.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_enc (TrialHandler)
        trials_enc.addData('key_resp_congruency.keys',key_resp_congruency.keys)
        trials_enc.addData('key_resp_congruency.corr', key_resp_congruency.corr)
        if key_resp_congruency.keys != None:  # we had a response
            trials_enc.addData('key_resp_congruency.rt', key_resp_congruency.rt)
            trials_enc.addData('key_resp_congruency.duration', key_resp_congruency.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if congruency_response.maxDurationReached:
            routineTimer.addTime(-congruency_response.maxDuration)
        elif congruency_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "blank1000_2" ---
        # create an object to store info about Routine blank1000_2
        blank1000_2 = data.Routine(
            name='blank1000_2',
            components=[text_8],
        )
        blank1000_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank1000_2
        blank1000_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank1000_2.tStart = globalClock.getTime(format='float')
        blank1000_2.status = STARTED
        thisExp.addData('blank1000_2.started', blank1000_2.tStart)
        blank1000_2.maxDuration = None
        # keep track of which components have finished
        blank1000_2Components = blank1000_2.components
        for thisComponent in blank1000_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank1000_2" ---
        thisExp.currentRoutine = blank1000_2
        blank1000_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_enc, 'status') and thisTrials_enc.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank1000_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank1000_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank1000_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank1000_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank1000_2" ---
        for thisComponent in blank1000_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank1000_2
        blank1000_2.tStop = globalClock.getTime(format='float')
        blank1000_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank1000_2.stopped', blank1000_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank1000_2.maxDurationReached:
            routineTimer.addTime(-blank1000_2.maxDuration)
        elif blank1000_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisTrials_enc as finished
        if hasattr(thisTrials_enc, 'status'):
            thisTrials_enc.status = FINISHED
        # if awaiting a pause, pause now
        if trials_enc.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_enc.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_enc'
    trials_enc.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "ret_inst_gen" ---
    # create an object to store info about Routine ret_inst_gen
    ret_inst_gen = data.Routine(
        name='ret_inst_gen',
        components=[text_9, key_continue_7],
    )
    ret_inst_gen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_7
    key_continue_7.keys = []
    key_continue_7.rt = []
    _key_continue_7_allKeys = []
    # store start times for ret_inst_gen
    ret_inst_gen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_gen.tStart = globalClock.getTime(format='float')
    ret_inst_gen.status = STARTED
    thisExp.addData('ret_inst_gen.started', ret_inst_gen.tStart)
    ret_inst_gen.maxDuration = None
    # keep track of which components have finished
    ret_inst_genComponents = ret_inst_gen.components
    for thisComponent in ret_inst_gen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_gen" ---
    thisExp.currentRoutine = ret_inst_gen
    ret_inst_gen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_9.started')
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *key_continue_7* updates
        waitOnFlip = False
        
        # if key_continue_7 is starting this frame...
        if key_continue_7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_continue_7.frameNStart = frameN  # exact frame index
            key_continue_7.tStart = t  # local t and not account for scr refresh
            key_continue_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_7.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_7_allKeys.extend(theseKeys)
            if len(_key_continue_7_allKeys):
                key_continue_7.keys = _key_continue_7_allKeys[-1].name  # just the last key pressed
                key_continue_7.rt = _key_continue_7_allKeys[-1].rt
                key_continue_7.duration = _key_continue_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_gen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_gen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_gen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_gen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_gen" ---
    for thisComponent in ret_inst_gen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_gen
    ret_inst_gen.tStop = globalClock.getTime(format='float')
    ret_inst_gen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_gen.stopped', ret_inst_gen.tStop)
    # check responses
    if key_continue_7.keys in ['', [], None]:  # No response was made
        key_continue_7.keys = None
    thisExp.addData('key_continue_7.keys',key_continue_7.keys)
    if key_continue_7.keys != None:  # we had a response
        thisExp.addData('key_continue_7.rt', key_continue_7.rt)
        thisExp.addData('key_continue_7.duration', key_continue_7.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_gen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_1" ---
    # create an object to store info about Routine ret_inst_1
    ret_inst_1 = data.Routine(
        name='ret_inst_1',
        components=[],
    )
    ret_inst_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_ret_rand_vars
    counterbalance_file = 'counterbalanced_vars_behav.csv'
    
    df_cb = pd.read_csv(counterbalance_file, sep=None, engine='python', skiprows=1)
    
    # Clean column names
    df_cb.columns = df_cb.columns.str.strip()
    df_cb.columns = df_cb.columns.str.replace('\ufeff', '')
    
    df_cb['Participant'] = df_cb['Participant'].astype(int)
    
    subnum = int(expInfo['participant'])
    
    matches = df_cb[df_cb['Participant'] == subnum]
    
    if len(matches) == 0:
        raise Exception(f"Participant {subnum} not found.")
    
    ind = matches.index[0]
    
    sub_conds = df_cb.loc[ind, df_cb.columns].values.tolist()
    
    # =========================================
    # CONTEXT KEYS
    # =========================================
    
    if sub_conds[2] == 0:
        key_jungle = '1'
        key_sea = '2'
    else:
        key_jungle = '2'
        key_sea = '1'
    
    # =========================================
    # COLORS
    # =========================================
    
    if sub_conds[3] == 0:
        color_jungle = 'purple'
        color_sea = 'pink'
    else:
        color_jungle = 'pink'
        color_sea = 'purple'
    
    # =========================================
    # SCENE KEYS
    # =========================================
    
    if sub_conds[4] == 0:
        key_JA = '1'
        key_JB = '2'
    else:
        key_JA = '2'
        key_JB = '1'
    
    if sub_conds[5] == 0:
        key_UA = '1'
        key_UB = '2'
    else:
        key_UA = '2'
        key_UB = '1'
    
    # =========================================
    # CONTEXT DISPLAY TEXT
    # =========================================
    
    if key_jungle == '1':
        context_jungle_txt = '1. Jungle'
        context_sea_txt = '2. Undersea'
    
        pos_jungle = (0,0.06)
        pos_sea = (0,0)
    
    else:
        context_sea_txt = '1. Undersea'
        context_jungle_txt = '2. Jungle'
    
        pos_sea = (0,0.06)
        pos_jungle = (0,0)
    
    print("TRAINING key_jungle:", key_jungle)
    print("TRAINING color_jungle:", color_jungle)
    print("TRAINING color_sea:", color_sea)
    print("DEBUG scene_stimulus:", scene_stimulus)
    print("DEBUG qst_num:", qst_num)
    print("DEBUG skipTrial:", skipTrial)
    # Run 'Begin Routine' code from code_ret_context_screen
    if key_jungle == '1':
        context_jungle_txt = '1. Jungle'
        pos_jungle = (0,0.06)
        context_sea_txt = '2. Undersea'
        pos_sea = (0,0)
    elif key_jungle == '2':
        context_sea_txt = '1. Undersea'
        pos_sea = (0,0.06)
        context_jungle_txt = '2. Jungle'
        pos_jungle = (0,0)
    # Run 'Begin Routine' code from code_ret_scene_screens
    # load the participant's scene-name mapping
    cond_file = "stimuli/cond_lists/scene_names_sub_" + expInfo['participant'] + ".csv"
    df = pd.read_csv(cond_file)
    
    # build dictionary mapping scene code -> label
    labels = dict(zip(df['scene_stimulus'], df['scene_label']))
    
    # Jungle scene labels
    if key_JA == '1':
        jungle_lbl_text = '1.' + labels['JA'] + '\n2.' + labels['JB']
    else:
        jungle_lbl_text = '1.' + labels['JB'] + '\n2.' + labels['JA']
    
    # Undersea scene labels
    if key_UA == '1':
        sea_lbl_text = '1.' + labels['UA'] + '\n2.' + labels['UB']
    else:
        sea_lbl_text = '1.' + labels['UB'] + '\n2.' + labels['UA']
    # store start times for ret_inst_1
    ret_inst_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_1.tStart = globalClock.getTime(format='float')
    ret_inst_1.status = STARTED
    thisExp.addData('ret_inst_1.started', ret_inst_1.tStart)
    ret_inst_1.maxDuration = None
    # keep track of which components have finished
    ret_inst_1Components = ret_inst_1.components
    for thisComponent in ret_inst_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_1" ---
    thisExp.currentRoutine = ret_inst_1
    ret_inst_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_1" ---
    for thisComponent in ret_inst_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_1
    ret_inst_1.tStop = globalClock.getTime(format='float')
    ret_inst_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_1.stopped', ret_inst_1.tStop)
    thisExp.nextEntry()
    # the Routine "ret_inst_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_2" ---
    # create an object to store info about Routine ret_inst_2
    ret_inst_2 = data.Routine(
        name='ret_inst_2',
        components=[image_inst2, key_continue_2],
    )
    ret_inst_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_2
    key_continue_2.keys = []
    key_continue_2.rt = []
    _key_continue_2_allKeys = []
    # store start times for ret_inst_2
    ret_inst_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_2.tStart = globalClock.getTime(format='float')
    ret_inst_2.status = STARTED
    thisExp.addData('ret_inst_2.started', ret_inst_2.tStart)
    ret_inst_2.maxDuration = None
    # keep track of which components have finished
    ret_inst_2Components = ret_inst_2.components
    for thisComponent in ret_inst_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_2" ---
    thisExp.currentRoutine = ret_inst_2
    ret_inst_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst2* updates
        
        # if image_inst2 is starting this frame...
        if image_inst2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst2.frameNStart = frameN  # exact frame index
            image_inst2.tStart = t  # local t and not account for scr refresh
            image_inst2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst2.started')
            # update status
            image_inst2.status = STARTED
            image_inst2.setAutoDraw(True)
        
        # if image_inst2 is active this frame...
        if image_inst2.status == STARTED:
            # update params
            pass
        
        # *key_continue_2* updates
        waitOnFlip = False
        
        # if key_continue_2 is starting this frame...
        if key_continue_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            key_continue_2.frameNStart = frameN  # exact frame index
            key_continue_2.tStart = t  # local t and not account for scr refresh
            key_continue_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_2.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_2_allKeys.extend(theseKeys)
            if len(_key_continue_2_allKeys):
                key_continue_2.keys = _key_continue_2_allKeys[-1].name  # just the last key pressed
                key_continue_2.rt = _key_continue_2_allKeys[-1].rt
                key_continue_2.duration = _key_continue_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_2" ---
    for thisComponent in ret_inst_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_2
    ret_inst_2.tStop = globalClock.getTime(format='float')
    ret_inst_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_2.stopped', ret_inst_2.tStop)
    # check responses
    if key_continue_2.keys in ['', [], None]:  # No response was made
        key_continue_2.keys = None
    thisExp.addData('key_continue_2.keys',key_continue_2.keys)
    if key_continue_2.keys != None:  # we had a response
        thisExp.addData('key_continue_2.rt', key_continue_2.rt)
        thisExp.addData('key_continue_2.duration', key_continue_2.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_3" ---
    # create an object to store info about Routine ret_inst_3
    ret_inst_3 = data.Routine(
        name='ret_inst_3',
        components=[image_inst3, key_continue_3],
    )
    ret_inst_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_3
    key_continue_3.keys = []
    key_continue_3.rt = []
    _key_continue_3_allKeys = []
    # store start times for ret_inst_3
    ret_inst_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_3.tStart = globalClock.getTime(format='float')
    ret_inst_3.status = STARTED
    thisExp.addData('ret_inst_3.started', ret_inst_3.tStart)
    ret_inst_3.maxDuration = None
    # keep track of which components have finished
    ret_inst_3Components = ret_inst_3.components
    for thisComponent in ret_inst_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_3" ---
    thisExp.currentRoutine = ret_inst_3
    ret_inst_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst3* updates
        
        # if image_inst3 is starting this frame...
        if image_inst3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst3.frameNStart = frameN  # exact frame index
            image_inst3.tStart = t  # local t and not account for scr refresh
            image_inst3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst3.started')
            # update status
            image_inst3.status = STARTED
            image_inst3.setAutoDraw(True)
        
        # if image_inst3 is active this frame...
        if image_inst3.status == STARTED:
            # update params
            pass
        
        # *key_continue_3* updates
        waitOnFlip = False
        
        # if key_continue_3 is starting this frame...
        if key_continue_3.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue_3.frameNStart = frameN  # exact frame index
            key_continue_3.tStart = t  # local t and not account for scr refresh
            key_continue_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_3.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_3_allKeys.extend(theseKeys)
            if len(_key_continue_3_allKeys):
                key_continue_3.keys = _key_continue_3_allKeys[-1].name  # just the last key pressed
                key_continue_3.rt = _key_continue_3_allKeys[-1].rt
                key_continue_3.duration = _key_continue_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_3" ---
    for thisComponent in ret_inst_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_3
    ret_inst_3.tStop = globalClock.getTime(format='float')
    ret_inst_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_3.stopped', ret_inst_3.tStop)
    # check responses
    if key_continue_3.keys in ['', [], None]:  # No response was made
        key_continue_3.keys = None
    thisExp.addData('key_continue_3.keys',key_continue_3.keys)
    if key_continue_3.keys != None:  # we had a response
        thisExp.addData('key_continue_3.rt', key_continue_3.rt)
        thisExp.addData('key_continue_3.duration', key_continue_3.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_4" ---
    # create an object to store info about Routine ret_inst_4
    ret_inst_4 = data.Routine(
        name='ret_inst_4',
        components=[image_inst4, key_continue_4],
    )
    ret_inst_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_4
    key_continue_4.keys = []
    key_continue_4.rt = []
    _key_continue_4_allKeys = []
    # store start times for ret_inst_4
    ret_inst_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_4.tStart = globalClock.getTime(format='float')
    ret_inst_4.status = STARTED
    thisExp.addData('ret_inst_4.started', ret_inst_4.tStart)
    ret_inst_4.maxDuration = None
    # keep track of which components have finished
    ret_inst_4Components = ret_inst_4.components
    for thisComponent in ret_inst_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_4" ---
    thisExp.currentRoutine = ret_inst_4
    ret_inst_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst4* updates
        
        # if image_inst4 is starting this frame...
        if image_inst4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst4.frameNStart = frameN  # exact frame index
            image_inst4.tStart = t  # local t and not account for scr refresh
            image_inst4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst4.started')
            # update status
            image_inst4.status = STARTED
            image_inst4.setAutoDraw(True)
        
        # if image_inst4 is active this frame...
        if image_inst4.status == STARTED:
            # update params
            pass
        
        # *key_continue_4* updates
        waitOnFlip = False
        
        # if key_continue_4 is starting this frame...
        if key_continue_4.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue_4.frameNStart = frameN  # exact frame index
            key_continue_4.tStart = t  # local t and not account for scr refresh
            key_continue_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_4.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_4_allKeys.extend(theseKeys)
            if len(_key_continue_4_allKeys):
                key_continue_4.keys = _key_continue_4_allKeys[-1].name  # just the last key pressed
                key_continue_4.rt = _key_continue_4_allKeys[-1].rt
                key_continue_4.duration = _key_continue_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_4,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_4.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_4.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_4" ---
    for thisComponent in ret_inst_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_4
    ret_inst_4.tStop = globalClock.getTime(format='float')
    ret_inst_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_4.stopped', ret_inst_4.tStop)
    # check responses
    if key_continue_4.keys in ['', [], None]:  # No response was made
        key_continue_4.keys = None
    thisExp.addData('key_continue_4.keys',key_continue_4.keys)
    if key_continue_4.keys != None:  # we had a response
        thisExp.addData('key_continue_4.rt', key_continue_4.rt)
        thisExp.addData('key_continue_4.duration', key_continue_4.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_5" ---
    # create an object to store info about Routine ret_inst_5
    ret_inst_5 = data.Routine(
        name='ret_inst_5',
        components=[image_inst5, key_continue_5],
    )
    ret_inst_5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_5
    key_continue_5.keys = []
    key_continue_5.rt = []
    _key_continue_5_allKeys = []
    # store start times for ret_inst_5
    ret_inst_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_5.tStart = globalClock.getTime(format='float')
    ret_inst_5.status = STARTED
    thisExp.addData('ret_inst_5.started', ret_inst_5.tStart)
    ret_inst_5.maxDuration = None
    # keep track of which components have finished
    ret_inst_5Components = ret_inst_5.components
    for thisComponent in ret_inst_5.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_5" ---
    thisExp.currentRoutine = ret_inst_5
    ret_inst_5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst5* updates
        
        # if image_inst5 is starting this frame...
        if image_inst5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst5.frameNStart = frameN  # exact frame index
            image_inst5.tStart = t  # local t and not account for scr refresh
            image_inst5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst5.started')
            # update status
            image_inst5.status = STARTED
            image_inst5.setAutoDraw(True)
        
        # if image_inst5 is active this frame...
        if image_inst5.status == STARTED:
            # update params
            pass
        
        # *key_continue_5* updates
        waitOnFlip = False
        
        # if key_continue_5 is starting this frame...
        if key_continue_5.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue_5.frameNStart = frameN  # exact frame index
            key_continue_5.tStart = t  # local t and not account for scr refresh
            key_continue_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_5.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_5_allKeys.extend(theseKeys)
            if len(_key_continue_5_allKeys):
                key_continue_5.keys = _key_continue_5_allKeys[-1].name  # just the last key pressed
                key_continue_5.rt = _key_continue_5_allKeys[-1].rt
                key_continue_5.duration = _key_continue_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_5,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_5.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_5.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_5" ---
    for thisComponent in ret_inst_5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_5
    ret_inst_5.tStop = globalClock.getTime(format='float')
    ret_inst_5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_5.stopped', ret_inst_5.tStop)
    # check responses
    if key_continue_5.keys in ['', [], None]:  # No response was made
        key_continue_5.keys = None
    thisExp.addData('key_continue_5.keys',key_continue_5.keys)
    if key_continue_5.keys != None:  # we had a response
        thisExp.addData('key_continue_5.rt', key_continue_5.rt)
        thisExp.addData('key_continue_5.duration', key_continue_5.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_6" ---
    # create an object to store info about Routine ret_inst_6
    ret_inst_6 = data.Routine(
        name='ret_inst_6',
        components=[image_inst6, key_continue_6],
    )
    ret_inst_6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue_6
    key_continue_6.keys = []
    key_continue_6.rt = []
    _key_continue_6_allKeys = []
    # store start times for ret_inst_6
    ret_inst_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_6.tStart = globalClock.getTime(format='float')
    ret_inst_6.status = STARTED
    thisExp.addData('ret_inst_6.started', ret_inst_6.tStart)
    ret_inst_6.maxDuration = None
    # keep track of which components have finished
    ret_inst_6Components = ret_inst_6.components
    for thisComponent in ret_inst_6.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_6" ---
    thisExp.currentRoutine = ret_inst_6
    ret_inst_6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst6* updates
        
        # if image_inst6 is starting this frame...
        if image_inst6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst6.frameNStart = frameN  # exact frame index
            image_inst6.tStart = t  # local t and not account for scr refresh
            image_inst6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst6.started')
            # update status
            image_inst6.status = STARTED
            image_inst6.setAutoDraw(True)
        
        # if image_inst6 is active this frame...
        if image_inst6.status == STARTED:
            # update params
            pass
        
        # *key_continue_6* updates
        waitOnFlip = False
        
        # if key_continue_6 is starting this frame...
        if key_continue_6.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue_6.frameNStart = frameN  # exact frame index
            key_continue_6.tStart = t  # local t and not account for scr refresh
            key_continue_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue_6.status == STARTED and not waitOnFlip:
            theseKeys = key_continue_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_6_allKeys.extend(theseKeys)
            if len(_key_continue_6_allKeys):
                key_continue_6.keys = _key_continue_6_allKeys[-1].name  # just the last key pressed
                key_continue_6.rt = _key_continue_6_allKeys[-1].rt
                key_continue_6.duration = _key_continue_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_6,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_6.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_6.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_6" ---
    for thisComponent in ret_inst_6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_6
    ret_inst_6.tStop = globalClock.getTime(format='float')
    ret_inst_6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_6.stopped', ret_inst_6.tStop)
    # check responses
    if key_continue_6.keys in ['', [], None]:  # No response was made
        key_continue_6.keys = None
    thisExp.addData('key_continue_6.keys',key_continue_6.keys)
    if key_continue_6.keys != None:  # we had a response
        thisExp.addData('key_continue_6.rt', key_continue_6.rt)
        thisExp.addData('key_continue_6.duration', key_continue_6.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_7" ---
    # create an object to store info about Routine ret_inst_7
    ret_inst_7 = data.Routine(
        name='ret_inst_7',
        components=[image_inst7, key_continue],
    )
    ret_inst_7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue
    key_continue.keys = []
    key_continue.rt = []
    _key_continue_allKeys = []
    # store start times for ret_inst_7
    ret_inst_7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_7.tStart = globalClock.getTime(format='float')
    ret_inst_7.status = STARTED
    thisExp.addData('ret_inst_7.started', ret_inst_7.tStart)
    ret_inst_7.maxDuration = None
    # keep track of which components have finished
    ret_inst_7Components = ret_inst_7.components
    for thisComponent in ret_inst_7.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ret_inst_7" ---
    thisExp.currentRoutine = ret_inst_7
    ret_inst_7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_inst7* updates
        
        # if image_inst7 is starting this frame...
        if image_inst7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_inst7.frameNStart = frameN  # exact frame index
            image_inst7.tStart = t  # local t and not account for scr refresh
            image_inst7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_inst7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_inst7.started')
            # update status
            image_inst7.status = STARTED
            image_inst7.setAutoDraw(True)
        
        # if image_inst7 is active this frame...
        if image_inst7.status == STARTED:
            # update params
            pass
        
        # *key_continue* updates
        waitOnFlip = False
        
        # if key_continue is starting this frame...
        if key_continue.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue.frameNStart = frameN  # exact frame index
            key_continue.tStart = t  # local t and not account for scr refresh
            key_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue.status == STARTED and not waitOnFlip:
            theseKeys = key_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_allKeys.extend(theseKeys)
            if len(_key_continue_allKeys):
                key_continue.keys = _key_continue_allKeys[-1].name  # just the last key pressed
                key_continue.rt = _key_continue_allKeys[-1].rt
                key_continue.duration = _key_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=ret_inst_7,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_7.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_7.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_7" ---
    for thisComponent in ret_inst_7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_7
    ret_inst_7.tStop = globalClock.getTime(format='float')
    ret_inst_7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_7.stopped', ret_inst_7.tStop)
    # check responses
    if key_continue.keys in ['', [], None]:  # No response was made
        key_continue.keys = None
    thisExp.addData('key_continue.keys',key_continue.keys)
    if key_continue.keys != None:  # we had a response
        thisExp.addData('key_continue.rt', key_continue.rt)
        thisExp.addData('key_continue.duration', key_continue.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    part4_loop = data.TrialHandler2(
        name='part4_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(stim_list), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(part4_loop)  # add the loop to the experiment
    thisPart4_loop = part4_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPart4_loop.rgb)
    if thisPart4_loop != None:
        for paramName in thisPart4_loop:
            globals()[paramName] = thisPart4_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPart4_loop in part4_loop:
        part4_loop.status = STARTED
        if hasattr(thisPart4_loop, 'status'):
            thisPart4_loop.status = STARTED
        currentLoop = part4_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPart4_loop.rgb)
        if thisPart4_loop != None:
            for paramName in thisPart4_loop:
                globals()[paramName] = thisPart4_loop[paramName]
        
        # --- Prepare to start Routine "item_imagine_scene" ---
        # create an object to store info about Routine item_imagine_scene
        item_imagine_scene = data.Routine(
            name='item_imagine_scene',
            components=[text_imagine, item_cue],
        )
        item_imagine_scene.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        item_cue.setImage(recalled_stimulus)
        # Run 'Begin Routine' code from code_15
        _ = visual.ImageStim(win, image=recalled_stimulus)
        # store start times for item_imagine_scene
        item_imagine_scene.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        item_imagine_scene.tStart = globalClock.getTime(format='float')
        item_imagine_scene.status = STARTED
        thisExp.addData('item_imagine_scene.started', item_imagine_scene.tStart)
        item_imagine_scene.maxDuration = None
        # keep track of which components have finished
        item_imagine_sceneComponents = item_imagine_scene.components
        for thisComponent in item_imagine_scene.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "item_imagine_scene" ---
        thisExp.currentRoutine = item_imagine_scene
        item_imagine_scene.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_imagine* updates
            
            # if text_imagine is starting this frame...
            if text_imagine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_imagine.frameNStart = frameN  # exact frame index
                text_imagine.tStart = t  # local t and not account for scr refresh
                text_imagine.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_imagine, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_imagine.status = STARTED
                text_imagine.setAutoDraw(True)
            
            # if text_imagine is active this frame...
            if text_imagine.status == STARTED:
                # update params
                pass
            
            # if text_imagine is stopping this frame...
            if text_imagine.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_imagine.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_imagine.tStop = t  # not accounting for scr refresh
                    text_imagine.tStopRefresh = tThisFlipGlobal  # on global time
                    text_imagine.frameNStop = frameN  # exact frame index
                    # update status
                    text_imagine.status = FINISHED
                    text_imagine.setAutoDraw(False)
            
            # *item_cue* updates
            
            # if item_cue is starting this frame...
            if item_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                item_cue.frameNStart = frameN  # exact frame index
                item_cue.tStart = t  # local t and not account for scr refresh
                item_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item_cue.started')
                # update status
                item_cue.status = STARTED
                item_cue.setAutoDraw(True)
            
            # if item_cue is active this frame...
            if item_cue.status == STARTED:
                # update params
                pass
            
            # if item_cue is stopping this frame...
            if item_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item_cue.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    item_cue.tStop = t  # not accounting for scr refresh
                    item_cue.tStopRefresh = tThisFlipGlobal  # on global time
                    item_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item_cue.stopped')
                    # update status
                    item_cue.status = FINISHED
                    item_cue.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=item_imagine_scene,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                item_imagine_scene.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if item_imagine_scene.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in item_imagine_scene.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "item_imagine_scene" ---
        for thisComponent in item_imagine_scene.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for item_imagine_scene
        item_imagine_scene.tStop = globalClock.getTime(format='float')
        item_imagine_scene.tStopRefresh = tThisFlipGlobal
        thisExp.addData('item_imagine_scene.stopped', item_imagine_scene.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if item_imagine_scene.maxDurationReached:
            routineTimer.addTime(-item_imagine_scene.maxDuration)
        elif item_imagine_scene.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "p4_blank1000" ---
        # create an object to store info about Routine p4_blank1000
        p4_blank1000 = data.Routine(
            name='p4_blank1000',
            components=[text_10],
        )
        p4_blank1000.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for p4_blank1000
        p4_blank1000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p4_blank1000.tStart = globalClock.getTime(format='float')
        p4_blank1000.status = STARTED
        thisExp.addData('p4_blank1000.started', p4_blank1000.tStart)
        p4_blank1000.maxDuration = None
        # keep track of which components have finished
        p4_blank1000Components = p4_blank1000.components
        for thisComponent in p4_blank1000.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p4_blank1000" ---
        thisExp.currentRoutine = p4_blank1000
        p4_blank1000.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_10* updates
            
            # if text_10 is starting this frame...
            if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_10.frameNStart = frameN  # exact frame index
                text_10.tStart = t  # local t and not account for scr refresh
                text_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_10.started')
                # update status
                text_10.status = STARTED
                text_10.setAutoDraw(True)
            
            # if text_10 is active this frame...
            if text_10.status == STARTED:
                # update params
                pass
            
            # if text_10 is stopping this frame...
            if text_10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_10.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text_10.tStop = t  # not accounting for scr refresh
                    text_10.tStopRefresh = tThisFlipGlobal  # on global time
                    text_10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_10.stopped')
                    # update status
                    text_10.status = FINISHED
                    text_10.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p4_blank1000,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p4_blank1000.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p4_blank1000.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p4_blank1000.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p4_blank1000" ---
        for thisComponent in p4_blank1000.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p4_blank1000
        p4_blank1000.tStop = globalClock.getTime(format='float')
        p4_blank1000.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p4_blank1000.stopped', p4_blank1000.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if p4_blank1000.maxDurationReached:
            routineTimer.addTime(-p4_blank1000.maxDuration)
        elif p4_blank1000.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "select_context" ---
        # create an object to store info about Routine select_context
        select_context = data.Routine(
            name='select_context',
            components=[text_contrext_headline, text_sea, text_jungle, key_resp_context],
        )
        select_context.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_cor_context_key
        # get the correct context for current trial
        
        if scene_cat == 'Jungle':
            cor_key = key_jungle
        elif scene_cat == 'Undersea':
            cor_key = key_sea
        text_sea.setColor(color_sea, colorSpace='rgb')
        text_sea.setPos(pos_sea)
        text_sea.setText(context_sea_txt)
        text_jungle.setColor(color_jungle, colorSpace='rgb')
        text_jungle.setPos(pos_jungle)
        text_jungle.setText(context_jungle_txt)
        # create starting attributes for key_resp_context
        key_resp_context.keys = []
        key_resp_context.rt = []
        _key_resp_context_allKeys = []
        # store start times for select_context
        select_context.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        select_context.tStart = globalClock.getTime(format='float')
        select_context.status = STARTED
        thisExp.addData('select_context.started', select_context.tStart)
        select_context.maxDuration = None
        # keep track of which components have finished
        select_contextComponents = select_context.components
        for thisComponent in select_context.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "select_context" ---
        thisExp.currentRoutine = select_context
        select_context.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_contrext_headline* updates
            
            # if text_contrext_headline is starting this frame...
            if text_contrext_headline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_contrext_headline.frameNStart = frameN  # exact frame index
                text_contrext_headline.tStart = t  # local t and not account for scr refresh
                text_contrext_headline.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_contrext_headline, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_contrext_headline.started')
                # update status
                text_contrext_headline.status = STARTED
                text_contrext_headline.setAutoDraw(True)
            
            # if text_contrext_headline is active this frame...
            if text_contrext_headline.status == STARTED:
                # update params
                pass
            
            # if text_contrext_headline is stopping this frame...
            if text_contrext_headline.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_contrext_headline.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_contrext_headline.tStop = t  # not accounting for scr refresh
                    text_contrext_headline.tStopRefresh = tThisFlipGlobal  # on global time
                    text_contrext_headline.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_contrext_headline.stopped')
                    # update status
                    text_contrext_headline.status = FINISHED
                    text_contrext_headline.setAutoDraw(False)
            
            # *text_sea* updates
            
            # if text_sea is starting this frame...
            if text_sea.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_sea.frameNStart = frameN  # exact frame index
                text_sea.tStart = t  # local t and not account for scr refresh
                text_sea.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_sea, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_sea.started')
                # update status
                text_sea.status = STARTED
                text_sea.setAutoDraw(True)
            
            # if text_sea is active this frame...
            if text_sea.status == STARTED:
                # update params
                pass
            
            # if text_sea is stopping this frame...
            if text_sea.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_sea.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_sea.tStop = t  # not accounting for scr refresh
                    text_sea.tStopRefresh = tThisFlipGlobal  # on global time
                    text_sea.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_sea.stopped')
                    # update status
                    text_sea.status = FINISHED
                    text_sea.setAutoDraw(False)
            
            # *text_jungle* updates
            
            # if text_jungle is starting this frame...
            if text_jungle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_jungle.frameNStart = frameN  # exact frame index
                text_jungle.tStart = t  # local t and not account for scr refresh
                text_jungle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_jungle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_jungle.started')
                # update status
                text_jungle.status = STARTED
                text_jungle.setAutoDraw(True)
            
            # if text_jungle is active this frame...
            if text_jungle.status == STARTED:
                # update params
                pass
            
            # if text_jungle is stopping this frame...
            if text_jungle.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_jungle.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_jungle.tStop = t  # not accounting for scr refresh
                    text_jungle.tStopRefresh = tThisFlipGlobal  # on global time
                    text_jungle.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_jungle.stopped')
                    # update status
                    text_jungle.status = FINISHED
                    text_jungle.setAutoDraw(False)
            
            # *key_resp_context* updates
            
            # if key_resp_context is starting this frame...
            if key_resp_context.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_context.frameNStart = frameN  # exact frame index
                key_resp_context.tStart = t  # local t and not account for scr refresh
                key_resp_context.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_context, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('key_resp_context.started', t)
                # update status
                key_resp_context.status = STARTED
                # keyboard checking is just starting
                key_resp_context.clock.reset()  # now t=0
                key_resp_context.clearEvents(eventType='keyboard')
            
            # if key_resp_context is stopping this frame...
            if key_resp_context.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_context.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_context.tStop = t  # not accounting for scr refresh
                    key_resp_context.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_context.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('key_resp_context.stopped', t)
                    # update status
                    key_resp_context.status = FINISHED
                    key_resp_context.status = FINISHED
            if key_resp_context.status == STARTED:
                theseKeys = key_resp_context.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_context_allKeys.extend(theseKeys)
                if len(_key_resp_context_allKeys):
                    key_resp_context.keys = _key_resp_context_allKeys[-1].name  # just the last key pressed
                    key_resp_context.rt = _key_resp_context_allKeys[-1].rt
                    key_resp_context.duration = _key_resp_context_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_context.keys == str(cor_key)) or (key_resp_context.keys == cor_key):
                        key_resp_context.corr = 1
                    else:
                        key_resp_context.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=select_context,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                select_context.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if select_context.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in select_context.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "select_context" ---
        for thisComponent in select_context.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for select_context
        select_context.tStop = globalClock.getTime(format='float')
        select_context.tStopRefresh = tThisFlipGlobal
        thisExp.addData('select_context.stopped', select_context.tStop)
        # check responses
        if key_resp_context.keys in ['', [], None]:  # No response was made
            key_resp_context.keys = None
            # was no response the correct answer?!
            if str(cor_key).lower() == 'none':
               key_resp_context.corr = 1;  # correct non-response
            else:
               key_resp_context.corr = 0;  # failed to respond (incorrectly)
        # store data for part4_loop (TrialHandler)
        part4_loop.addData('key_resp_context.keys',key_resp_context.keys)
        part4_loop.addData('key_resp_context.corr', key_resp_context.corr)
        if key_resp_context.keys != None:  # we had a response
            part4_loop.addData('key_resp_context.rt', key_resp_context.rt)
            part4_loop.addData('key_resp_context.duration', key_resp_context.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if select_context.maxDurationReached:
            routineTimer.addTime(-select_context.maxDuration)
        elif select_context.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[text_blank],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        thisExp.currentRoutine = blank500
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_blank* updates
            
            # if text_blank is starting this frame...
            if text_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blank.frameNStart = frameN  # exact frame index
                text_blank.tStart = t  # local t and not account for scr refresh
                text_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blank.started')
                # update status
                text_blank.status = STARTED
                text_blank.setAutoDraw(True)
            
            # if text_blank is active this frame...
            if text_blank.status == STARTED:
                # update params
                pass
            
            # if text_blank is stopping this frame...
            if text_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_blank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_blank.tStop = t  # not accounting for scr refresh
                    text_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    text_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank.stopped')
                    # update status
                    text_blank.status = FINISHED
                    text_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank500,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank500.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank500.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "context_confidence" ---
        # create an object to store info about Routine context_confidence
        context_confidence = data.Routine(
            name='context_confidence',
            components=[text_context_choice, slider_context_conf, key_resp_context_conf],
        )
        context_confidence.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_get_context_choice
        # get the chosen context to present above confidence scales
        
        if key_resp_context.keys == key_jungle:
            chosen_cont = 'Jungle'
            chosen_cont_color = color_jungle
        
        elif key_resp_context.keys == key_sea:
            chosen_cont = 'Undersea'
            chosen_cont_color = color_sea
        
        elif key_resp_context.keys == None:
            chosen_cont = 'Nothing was chosen'
            chosen_cont_color = 'Black'
        text_context_choice.setColor(chosen_cont_color, colorSpace='rgb')
        text_context_choice.setText(chosen_cont)
        slider_context_conf.reset()
        # create starting attributes for key_resp_context_conf
        key_resp_context_conf.keys = []
        key_resp_context_conf.rt = []
        _key_resp_context_conf_allKeys = []
        # store start times for context_confidence
        context_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        context_confidence.tStart = globalClock.getTime(format='float')
        context_confidence.status = STARTED
        thisExp.addData('context_confidence.started', context_confidence.tStart)
        context_confidence.maxDuration = None
        # keep track of which components have finished
        context_confidenceComponents = context_confidence.components
        for thisComponent in context_confidence.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "context_confidence" ---
        thisExp.currentRoutine = context_confidence
        context_confidence.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_context_choice* updates
            
            # if text_context_choice is starting this frame...
            if text_context_choice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_context_choice.frameNStart = frameN  # exact frame index
                text_context_choice.tStart = t  # local t and not account for scr refresh
                text_context_choice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_context_choice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_context_choice.started')
                # update status
                text_context_choice.status = STARTED
                text_context_choice.setAutoDraw(True)
            
            # if text_context_choice is active this frame...
            if text_context_choice.status == STARTED:
                # update params
                pass
            
            # if text_context_choice is stopping this frame...
            if text_context_choice.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_context_choice.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_context_choice.tStop = t  # not accounting for scr refresh
                    text_context_choice.tStopRefresh = tThisFlipGlobal  # on global time
                    text_context_choice.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_context_choice.stopped')
                    # update status
                    text_context_choice.status = FINISHED
                    text_context_choice.setAutoDraw(False)
            
            # *slider_context_conf* updates
            
            # if slider_context_conf is starting this frame...
            if slider_context_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_context_conf.frameNStart = frameN  # exact frame index
                slider_context_conf.tStart = t  # local t and not account for scr refresh
                slider_context_conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_context_conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_context_conf.started')
                # update status
                slider_context_conf.status = STARTED
                slider_context_conf.setAutoDraw(True)
            
            # if slider_context_conf is active this frame...
            if slider_context_conf.status == STARTED:
                # update params
                pass
            
            # if slider_context_conf is stopping this frame...
            if slider_context_conf.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > slider_context_conf.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    slider_context_conf.tStop = t  # not accounting for scr refresh
                    slider_context_conf.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_context_conf.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_context_conf.stopped')
                    # update status
                    slider_context_conf.status = FINISHED
                    slider_context_conf.setAutoDraw(False)
            
            # *key_resp_context_conf* updates
            waitOnFlip = False
            
            # if key_resp_context_conf is starting this frame...
            if key_resp_context_conf.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_context_conf.frameNStart = frameN  # exact frame index
                key_resp_context_conf.tStart = t  # local t and not account for scr refresh
                key_resp_context_conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_context_conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_context_conf.started')
                # update status
                key_resp_context_conf.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_context_conf.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_context_conf.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_context_conf is stopping this frame...
            if key_resp_context_conf.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_context_conf.tStartRefresh + 1.8-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_context_conf.tStop = t  # not accounting for scr refresh
                    key_resp_context_conf.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_context_conf.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_context_conf.stopped')
                    # update status
                    key_resp_context_conf.status = FINISHED
                    key_resp_context_conf.status = FINISHED
            if key_resp_context_conf.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_context_conf.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_context_conf_allKeys.extend(theseKeys)
                if len(_key_resp_context_conf_allKeys):
                    key_resp_context_conf.keys = _key_resp_context_conf_allKeys[-1].name  # just the last key pressed
                    key_resp_context_conf.rt = _key_resp_context_conf_allKeys[-1].rt
                    key_resp_context_conf.duration = _key_resp_context_conf_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=context_confidence,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                context_confidence.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if context_confidence.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in context_confidence.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "context_confidence" ---
        for thisComponent in context_confidence.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for context_confidence
        context_confidence.tStop = globalClock.getTime(format='float')
        context_confidence.tStopRefresh = tThisFlipGlobal
        thisExp.addData('context_confidence.stopped', context_confidence.tStop)
        # check responses
        if key_resp_context_conf.keys in ['', [], None]:  # No response was made
            key_resp_context_conf.keys = None
        part4_loop.addData('key_resp_context_conf.keys',key_resp_context_conf.keys)
        if key_resp_context_conf.keys != None:  # we had a response
            part4_loop.addData('key_resp_context_conf.rt', key_resp_context_conf.rt)
            part4_loop.addData('key_resp_context_conf.duration', key_resp_context_conf.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if context_confidence.maxDurationReached:
            routineTimer.addTime(-context_confidence.maxDuration)
        elif context_confidence.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "p4_blank1000" ---
        # create an object to store info about Routine p4_blank1000
        p4_blank1000 = data.Routine(
            name='p4_blank1000',
            components=[text_10],
        )
        p4_blank1000.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for p4_blank1000
        p4_blank1000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p4_blank1000.tStart = globalClock.getTime(format='float')
        p4_blank1000.status = STARTED
        thisExp.addData('p4_blank1000.started', p4_blank1000.tStart)
        p4_blank1000.maxDuration = None
        # keep track of which components have finished
        p4_blank1000Components = p4_blank1000.components
        for thisComponent in p4_blank1000.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p4_blank1000" ---
        thisExp.currentRoutine = p4_blank1000
        p4_blank1000.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_10* updates
            
            # if text_10 is starting this frame...
            if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_10.frameNStart = frameN  # exact frame index
                text_10.tStart = t  # local t and not account for scr refresh
                text_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_10.started')
                # update status
                text_10.status = STARTED
                text_10.setAutoDraw(True)
            
            # if text_10 is active this frame...
            if text_10.status == STARTED:
                # update params
                pass
            
            # if text_10 is stopping this frame...
            if text_10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_10.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text_10.tStop = t  # not accounting for scr refresh
                    text_10.tStopRefresh = tThisFlipGlobal  # on global time
                    text_10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_10.stopped')
                    # update status
                    text_10.status = FINISHED
                    text_10.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=p4_blank1000,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p4_blank1000.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p4_blank1000.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p4_blank1000.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p4_blank1000" ---
        for thisComponent in p4_blank1000.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p4_blank1000
        p4_blank1000.tStop = globalClock.getTime(format='float')
        p4_blank1000.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p4_blank1000.stopped', p4_blank1000.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if p4_blank1000.maxDurationReached:
            routineTimer.addTime(-p4_blank1000.maxDuration)
        elif p4_blank1000.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "select_scene_name" ---
        # create an object to store info about Routine select_scene_name
        select_scene_name = data.Routine(
            name='select_scene_name',
            components=[text_scene_headline, text_actual_context, text_q_scene, text_labels, key_resp_scenes],
        )
        select_scene_name.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_set_scene_vars
        
        # set current trial's scene labels, color and correct response key
        
        if scene_cat == 'Jungle':
            cur_labels = jungle_lbl_text
            cur_color = color_jungle
        
        elif scene_cat == 'Undersea':
            cur_labels = sea_lbl_text
            cur_color = color_sea
        
        cor_scene_key = None
        
        stim = integrated_stimulus.lower()
        
        if 'jungle_a' in stim:
            cor_scene_key = key_JA
        
        elif 'jungle_b' in stim:
            cor_scene_key = key_JB
        
        elif 'ocean_a' in stim:
            cor_scene_key = key_UA
        
        elif 'ocean_b' in stim:
            cor_scene_key = key_UB
        
        
        # DEBUG (keep this for now)
        if cor_scene_key is None:
            print("🚨 couldn't determine scene from filename")
        text_actual_context.setColor(cur_color, colorSpace='rgb')
        text_actual_context.setText(scene_cat)
        text_labels.setColor(cur_color, colorSpace='rgb')
        text_labels.setText(cur_labels)
        # create starting attributes for key_resp_scenes
        key_resp_scenes.keys = []
        key_resp_scenes.rt = []
        _key_resp_scenes_allKeys = []
        # store start times for select_scene_name
        select_scene_name.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        select_scene_name.tStart = globalClock.getTime(format='float')
        select_scene_name.status = STARTED
        thisExp.addData('select_scene_name.started', select_scene_name.tStart)
        select_scene_name.maxDuration = None
        # keep track of which components have finished
        select_scene_nameComponents = select_scene_name.components
        for thisComponent in select_scene_name.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "select_scene_name" ---
        thisExp.currentRoutine = select_scene_name
        select_scene_name.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_scene_headline* updates
            
            # if text_scene_headline is starting this frame...
            if text_scene_headline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_scene_headline.frameNStart = frameN  # exact frame index
                text_scene_headline.tStart = t  # local t and not account for scr refresh
                text_scene_headline.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_scene_headline, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_scene_headline.started')
                # update status
                text_scene_headline.status = STARTED
                text_scene_headline.setAutoDraw(True)
            
            # if text_scene_headline is active this frame...
            if text_scene_headline.status == STARTED:
                # update params
                pass
            
            # if text_scene_headline is stopping this frame...
            if text_scene_headline.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_scene_headline.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_scene_headline.tStop = t  # not accounting for scr refresh
                    text_scene_headline.tStopRefresh = tThisFlipGlobal  # on global time
                    text_scene_headline.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_scene_headline.stopped')
                    # update status
                    text_scene_headline.status = FINISHED
                    text_scene_headline.setAutoDraw(False)
            
            # *text_actual_context* updates
            
            # if text_actual_context is starting this frame...
            if text_actual_context.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_actual_context.frameNStart = frameN  # exact frame index
                text_actual_context.tStart = t  # local t and not account for scr refresh
                text_actual_context.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_actual_context, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_actual_context.started')
                # update status
                text_actual_context.status = STARTED
                text_actual_context.setAutoDraw(True)
            
            # if text_actual_context is active this frame...
            if text_actual_context.status == STARTED:
                # update params
                pass
            
            # if text_actual_context is stopping this frame...
            if text_actual_context.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_actual_context.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_actual_context.tStop = t  # not accounting for scr refresh
                    text_actual_context.tStopRefresh = tThisFlipGlobal  # on global time
                    text_actual_context.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_actual_context.stopped')
                    # update status
                    text_actual_context.status = FINISHED
                    text_actual_context.setAutoDraw(False)
            
            # *text_q_scene* updates
            
            # if text_q_scene is starting this frame...
            if text_q_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_q_scene.frameNStart = frameN  # exact frame index
                text_q_scene.tStart = t  # local t and not account for scr refresh
                text_q_scene.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_q_scene, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_q_scene.started')
                # update status
                text_q_scene.status = STARTED
                text_q_scene.setAutoDraw(True)
            
            # if text_q_scene is active this frame...
            if text_q_scene.status == STARTED:
                # update params
                pass
            
            # if text_q_scene is stopping this frame...
            if text_q_scene.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_q_scene.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_q_scene.tStop = t  # not accounting for scr refresh
                    text_q_scene.tStopRefresh = tThisFlipGlobal  # on global time
                    text_q_scene.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_q_scene.stopped')
                    # update status
                    text_q_scene.status = FINISHED
                    text_q_scene.setAutoDraw(False)
            
            # *text_labels* updates
            
            # if text_labels is starting this frame...
            if text_labels.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_labels.frameNStart = frameN  # exact frame index
                text_labels.tStart = t  # local t and not account for scr refresh
                text_labels.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_labels, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_labels.started')
                # update status
                text_labels.status = STARTED
                text_labels.setAutoDraw(True)
            
            # if text_labels is active this frame...
            if text_labels.status == STARTED:
                # update params
                pass
            
            # if text_labels is stopping this frame...
            if text_labels.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_labels.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_labels.tStop = t  # not accounting for scr refresh
                    text_labels.tStopRefresh = tThisFlipGlobal  # on global time
                    text_labels.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_labels.stopped')
                    # update status
                    text_labels.status = FINISHED
                    text_labels.setAutoDraw(False)
            
            # *key_resp_scenes* updates
            waitOnFlip = False
            
            # if key_resp_scenes is starting this frame...
            if key_resp_scenes.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_scenes.frameNStart = frameN  # exact frame index
                key_resp_scenes.tStart = t  # local t and not account for scr refresh
                key_resp_scenes.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_scenes, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_scenes.started')
                # update status
                key_resp_scenes.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_scenes.clock.reset)  # t=0 on next screen flip
            
            # if key_resp_scenes is stopping this frame...
            if key_resp_scenes.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_scenes.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_scenes.tStop = t  # not accounting for scr refresh
                    key_resp_scenes.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_scenes.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_scenes.stopped')
                    # update status
                    key_resp_scenes.status = FINISHED
                    key_resp_scenes.status = FINISHED
            if key_resp_scenes.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_scenes.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_scenes_allKeys.extend(theseKeys)
                if len(_key_resp_scenes_allKeys):
                    key_resp_scenes.keys = _key_resp_scenes_allKeys[-1].name  # just the last key pressed
                    key_resp_scenes.rt = _key_resp_scenes_allKeys[-1].rt
                    key_resp_scenes.duration = _key_resp_scenes_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_scenes.keys == str(cor_scene_key)) or (key_resp_scenes.keys == cor_scene_key):
                        key_resp_scenes.corr = 1
                    else:
                        key_resp_scenes.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=select_scene_name,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                select_scene_name.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if select_scene_name.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in select_scene_name.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "select_scene_name" ---
        for thisComponent in select_scene_name.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for select_scene_name
        select_scene_name.tStop = globalClock.getTime(format='float')
        select_scene_name.tStopRefresh = tThisFlipGlobal
        thisExp.addData('select_scene_name.stopped', select_scene_name.tStop)
        # check responses
        if key_resp_scenes.keys in ['', [], None]:  # No response was made
            key_resp_scenes.keys = None
            # was no response the correct answer?!
            if str(cor_scene_key).lower() == 'none':
               key_resp_scenes.corr = 1;  # correct non-response
            else:
               key_resp_scenes.corr = 0;  # failed to respond (incorrectly)
        # store data for part4_loop (TrialHandler)
        part4_loop.addData('key_resp_scenes.keys',key_resp_scenes.keys)
        part4_loop.addData('key_resp_scenes.corr', key_resp_scenes.corr)
        if key_resp_scenes.keys != None:  # we had a response
            part4_loop.addData('key_resp_scenes.rt', key_resp_scenes.rt)
            part4_loop.addData('key_resp_scenes.duration', key_resp_scenes.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if select_scene_name.maxDurationReached:
            routineTimer.addTime(-select_scene_name.maxDuration)
        elif select_scene_name.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[text_blank],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        thisExp.currentRoutine = blank500
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_blank* updates
            
            # if text_blank is starting this frame...
            if text_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blank.frameNStart = frameN  # exact frame index
                text_blank.tStart = t  # local t and not account for scr refresh
                text_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blank.started')
                # update status
                text_blank.status = STARTED
                text_blank.setAutoDraw(True)
            
            # if text_blank is active this frame...
            if text_blank.status == STARTED:
                # update params
                pass
            
            # if text_blank is stopping this frame...
            if text_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_blank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_blank.tStop = t  # not accounting for scr refresh
                    text_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    text_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank.stopped')
                    # update status
                    text_blank.status = FINISHED
                    text_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank500,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank500.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank500.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "select_confidence" ---
        # create an object to store info about Routine select_confidence
        select_confidence = data.Routine(
            name='select_confidence',
            components=[text_chosen_scene, slider_scene_conf, key_resp_scene_conf],
        )
        select_confidence.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_get_chosen_scene
        # get the chosen scene to present above confidence scales
        
        if scene_cat == 'Jungle':
        
            if key_resp_scenes.keys == key_JA:
                chosen_scene = labels['JA']
        
            elif key_resp_scenes.keys == key_JB:
                chosen_scene = labels['JB']
        
            else:
                chosen_scene = 'Nothing was chosen'
        
        
        elif scene_cat == 'Undersea':
        
            if key_resp_scenes.keys == key_UA:
                chosen_scene = labels['UA']
        
            elif key_resp_scenes.keys == key_UB:
                chosen_scene = labels['UB']
        
            else:
                chosen_scene = 'Nothing was chosen'
        text_chosen_scene.setColor(cur_color, colorSpace='rgb')
        text_chosen_scene.setText(chosen_scene)
        slider_scene_conf.reset()
        # create starting attributes for key_resp_scene_conf
        key_resp_scene_conf.keys = []
        key_resp_scene_conf.rt = []
        _key_resp_scene_conf_allKeys = []
        # store start times for select_confidence
        select_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        select_confidence.tStart = globalClock.getTime(format='float')
        select_confidence.status = STARTED
        thisExp.addData('select_confidence.started', select_confidence.tStart)
        select_confidence.maxDuration = None
        # keep track of which components have finished
        select_confidenceComponents = select_confidence.components
        for thisComponent in select_confidence.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "select_confidence" ---
        thisExp.currentRoutine = select_confidence
        select_confidence.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_chosen_scene* updates
            
            # if text_chosen_scene is starting this frame...
            if text_chosen_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_chosen_scene.frameNStart = frameN  # exact frame index
                text_chosen_scene.tStart = t  # local t and not account for scr refresh
                text_chosen_scene.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_chosen_scene, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_chosen_scene.started')
                # update status
                text_chosen_scene.status = STARTED
                text_chosen_scene.setAutoDraw(True)
            
            # if text_chosen_scene is active this frame...
            if text_chosen_scene.status == STARTED:
                # update params
                pass
            
            # if text_chosen_scene is stopping this frame...
            if text_chosen_scene.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_chosen_scene.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_chosen_scene.tStop = t  # not accounting for scr refresh
                    text_chosen_scene.tStopRefresh = tThisFlipGlobal  # on global time
                    text_chosen_scene.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_chosen_scene.stopped')
                    # update status
                    text_chosen_scene.status = FINISHED
                    text_chosen_scene.setAutoDraw(False)
            
            # *slider_scene_conf* updates
            
            # if slider_scene_conf is starting this frame...
            if slider_scene_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_scene_conf.frameNStart = frameN  # exact frame index
                slider_scene_conf.tStart = t  # local t and not account for scr refresh
                slider_scene_conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_scene_conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_scene_conf.started')
                # update status
                slider_scene_conf.status = STARTED
                slider_scene_conf.setAutoDraw(True)
            
            # if slider_scene_conf is active this frame...
            if slider_scene_conf.status == STARTED:
                # update params
                pass
            
            # if slider_scene_conf is stopping this frame...
            if slider_scene_conf.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > slider_scene_conf.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    slider_scene_conf.tStop = t  # not accounting for scr refresh
                    slider_scene_conf.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_scene_conf.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_scene_conf.stopped')
                    # update status
                    slider_scene_conf.status = FINISHED
                    slider_scene_conf.setAutoDraw(False)
            
            # *key_resp_scene_conf* updates
            waitOnFlip = False
            
            # if key_resp_scene_conf is starting this frame...
            if key_resp_scene_conf.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_scene_conf.frameNStart = frameN  # exact frame index
                key_resp_scene_conf.tStart = t  # local t and not account for scr refresh
                key_resp_scene_conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_scene_conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_scene_conf.started')
                # update status
                key_resp_scene_conf.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_scene_conf.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_scene_conf.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_scene_conf is stopping this frame...
            if key_resp_scene_conf.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_scene_conf.tStartRefresh + 1.9-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_scene_conf.tStop = t  # not accounting for scr refresh
                    key_resp_scene_conf.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_scene_conf.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_scene_conf.stopped')
                    # update status
                    key_resp_scene_conf.status = FINISHED
                    key_resp_scene_conf.status = FINISHED
            if key_resp_scene_conf.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_scene_conf.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_scene_conf_allKeys.extend(theseKeys)
                if len(_key_resp_scene_conf_allKeys):
                    key_resp_scene_conf.keys = _key_resp_scene_conf_allKeys[-1].name  # just the last key pressed
                    key_resp_scene_conf.rt = _key_resp_scene_conf_allKeys[-1].rt
                    key_resp_scene_conf.duration = _key_resp_scene_conf_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=select_confidence,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                select_confidence.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if select_confidence.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in select_confidence.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "select_confidence" ---
        for thisComponent in select_confidence.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for select_confidence
        select_confidence.tStop = globalClock.getTime(format='float')
        select_confidence.tStopRefresh = tThisFlipGlobal
        thisExp.addData('select_confidence.stopped', select_confidence.tStop)
        # check responses
        if key_resp_scene_conf.keys in ['', [], None]:  # No response was made
            key_resp_scene_conf.keys = None
        part4_loop.addData('key_resp_scene_conf.keys',key_resp_scene_conf.keys)
        if key_resp_scene_conf.keys != None:  # we had a response
            part4_loop.addData('key_resp_scene_conf.rt', key_resp_scene_conf.rt)
            part4_loop.addData('key_resp_scene_conf.duration', key_resp_scene_conf.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if select_confidence.maxDurationReached:
            routineTimer.addTime(-select_confidence.maxDuration)
        elif select_confidence.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_3000" ---
        # create an object to store info about Routine blank_3000
        blank_3000 = data.Routine(
            name='blank_3000',
            components=[text_blank3000],
        )
        blank_3000.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_3000
        blank_3000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_3000.tStart = globalClock.getTime(format='float')
        blank_3000.status = STARTED
        thisExp.addData('blank_3000.started', blank_3000.tStart)
        blank_3000.maxDuration = None
        # keep track of which components have finished
        blank_3000Components = blank_3000.components
        for thisComponent in blank_3000.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_3000" ---
        thisExp.currentRoutine = blank_3000
        blank_3000.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart4_loop, 'status') and thisPart4_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_blank3000* updates
            
            # if text_blank3000 is starting this frame...
            if text_blank3000.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blank3000.frameNStart = frameN  # exact frame index
                text_blank3000.tStart = t  # local t and not account for scr refresh
                text_blank3000.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blank3000, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blank3000.started')
                # update status
                text_blank3000.status = STARTED
                text_blank3000.setAutoDraw(True)
            
            # if text_blank3000 is active this frame...
            if text_blank3000.status == STARTED:
                # update params
                pass
            
            # if text_blank3000 is stopping this frame...
            if text_blank3000.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_blank3000.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_blank3000.tStop = t  # not accounting for scr refresh
                    text_blank3000.tStopRefresh = tThisFlipGlobal  # on global time
                    text_blank3000.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank3000.stopped')
                    # update status
                    text_blank3000.status = FINISHED
                    text_blank3000.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_3000,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_3000.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_3000.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_3000.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_3000" ---
        for thisComponent in blank_3000.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_3000
        blank_3000.tStop = globalClock.getTime(format='float')
        blank_3000.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_3000.stopped', blank_3000.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_3000.maxDurationReached:
            routineTimer.addTime(-blank_3000.maxDuration)
        elif blank_3000.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        # mark thisPart4_loop as finished
        if hasattr(thisPart4_loop, 'status'):
            thisPart4_loop.status = FINISHED
        # if awaiting a pause, pause now
        if part4_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            part4_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'part4_loop'
    part4_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_screen" ---
    # create an object to store info about Routine end_screen
    end_screen = data.Routine(
        name='end_screen',
        components=[text_end_screen, key_resp_exit],
    )
    end_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_exit
    key_resp_exit.keys = []
    key_resp_exit.rt = []
    _key_resp_exit_allKeys = []
    # store start times for end_screen
    end_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_screen.tStart = globalClock.getTime(format='float')
    end_screen.status = STARTED
    thisExp.addData('end_screen.started', end_screen.tStart)
    end_screen.maxDuration = None
    # keep track of which components have finished
    end_screenComponents = end_screen.components
    for thisComponent in end_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_screen" ---
    thisExp.currentRoutine = end_screen
    end_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end_screen* updates
        
        # if text_end_screen is starting this frame...
        if text_end_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end_screen.frameNStart = frameN  # exact frame index
            text_end_screen.tStart = t  # local t and not account for scr refresh
            text_end_screen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end_screen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end_screen.started')
            # update status
            text_end_screen.status = STARTED
            text_end_screen.setAutoDraw(True)
        
        # if text_end_screen is active this frame...
        if text_end_screen.status == STARTED:
            # update params
            pass
        
        # if text_end_screen is stopping this frame...
        if text_end_screen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end_screen.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_end_screen.tStop = t  # not accounting for scr refresh
                text_end_screen.tStopRefresh = tThisFlipGlobal  # on global time
                text_end_screen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end_screen.stopped')
                # update status
                text_end_screen.status = FINISHED
                text_end_screen.setAutoDraw(False)
        
        # *key_resp_exit* updates
        waitOnFlip = False
        
        # if key_resp_exit is starting this frame...
        if key_resp_exit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_exit.frameNStart = frameN  # exact frame index
            key_resp_exit.tStart = t  # local t and not account for scr refresh
            key_resp_exit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_exit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_exit.started')
            # update status
            key_resp_exit.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_exit.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_exit.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_exit is stopping this frame...
        if key_resp_exit.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_exit.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_exit.tStop = t  # not accounting for scr refresh
                key_resp_exit.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_exit.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_exit.stopped')
                # update status
                key_resp_exit.status = FINISHED
                key_resp_exit.status = FINISHED
        if key_resp_exit.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_exit.getKeys(keyList=['z'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_exit_allKeys.extend(theseKeys)
            if len(_key_resp_exit_allKeys):
                key_resp_exit.keys = _key_resp_exit_allKeys[-1].name  # just the last key pressed
                key_resp_exit.rt = _key_resp_exit_allKeys[-1].rt
                key_resp_exit.duration = _key_resp_exit_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_screen" ---
    for thisComponent in end_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_screen
    end_screen.tStop = globalClock.getTime(format='float')
    end_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_screen.stopped', end_screen.tStop)
    # check responses
    if key_resp_exit.keys in ['', [], None]:  # No response was made
        key_resp_exit.keys = None
    thisExp.addData('key_resp_exit.keys',key_resp_exit.keys)
    if key_resp_exit.keys != None:  # we had a response
        thisExp.addData('key_resp_exit.rt', key_resp_exit.rt)
        thisExp.addData('key_resp_exit.duration', key_resp_exit.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end_screen.maxDurationReached:
        routineTimer.addTime(-end_screen.maxDuration)
    elif end_screen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
