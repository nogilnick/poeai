'''
Contains all constant values for the PoE AI
'''
#Cast times for different spells
QB_CT = 0.55
WB_CT = 0.19
EB_CT = 0.4
RB_CT = 0.44
TB_CT = 0.39
CT_LT = {'q':QB_CT, 'w':WB_CT, 'e':EB_CT, 'r':RB_CT, 't':TB_CT}
#Movement Map constants
MM_ISOP = 0.7   #Percantage of nearby closed cells for a cell to be considered isolated
MM_NSP = 16     #Number of points to sample getting nearby points
#Health and mana low values
HLOW = 0.75
MLOW = 0.25
MOVE_OKAY = 0       #Enumerated values for movement results
MOVE_FAIL = 1
MOVE_INPR = 2
MOVE_UNKW = 3
MOVE_NONE = 4
LW_TIMEOUT = 1.0 / 3.0
MOV_TO = 0.2
OBT_WT = 0.01       #Wait times for detection loops
ENT_WT = 0.01
LWT_WT = 0.01
#Number of prediction labels for obstacle CNN
NOPL = 2
#Prediction labels (Must be in sorted order for each network)
PL_C = 0
PL_O = 1
#Yes/No prediction labels
PL_N = 0
PL_Y = 1
#States of the bot
EXPL0 = 0     #First explore stage
EXPL1 = 1     #Second explore stage
EXPL2 = 2     #Third explore stage
ATCK0 = 3     #First attack stage
ATCK1 = 4     #Second attack stage
EVAD0 = 5     #Evade enemies
EVAD1 = 6
RETR0 = 7     #Retrace steps
HOME0 = 8     #Return to start
RMOV0 = 9     #Random movement
#Lookup of state to name
STN = ['EXPL0', 'EXPL1', 'EXPL2', 'ATCK0', 'ATCK1', 'EVAD0', 'EVAD1', 'RETR0', 'HOME0', 'RMOV0']
#State transition table
#E: Enemies detected; F: Failed movement; H: Low health; N: None
#      N      E     F      FE   H      H E    HF     HFE   M       M E   M F    M FE  MH     MH E   MHF    MHFE
STT = [
 [EXPL0, ATCK0, EXPL1, ATCK0, EVAD0, EVAD0, EVAD0, EVAD0, EXPL0, ATCK0,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#EXPL0 First explore stage
 [EXPL1, ATCK0, EXPL2, ATCK0, EVAD0, EVAD0, EVAD0, EVAD0, EXPL0, ATCK0,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#EXPL1 Second explore stage
 [EXPL2, ATCK0, RETR0, ATCK0, EVAD0, EVAD0, EVAD0, EVAD0, EXPL0, ATCK0,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#EXPL2 Third explore stage
 [EVAD0, ATCK1, EVAD0, ATCK1, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, ATCK1,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#ATCK0 First attack stage
 [EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0, EVAD0,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#ATCK1 Second attack stage
 [EVAD1, EVAD1, EVAD1, EVAD1, EVAD1, EVAD1, EVAD1, EVAD1, EVAD1, EVAD1,    -1,    -1, EVAD1, EVAD1,    -1,    -1],#EVAD0 Evade enemies
 [EVAD1, ATCK0, ATCK0, ATCK0, EVAD1, ATCK0, EVAD1, ATCK0, EVAD1, ATCK0,    -1,    -1, EVAD1, ATCK0,    -1,    -1],#EVAD1 Evade enemies
 [RETR0, ATCK0, RMOV0, ATCK0, EVAD0, EVAD0, EVAD0, EVAD0, EXPL0, ATCK0,    -1,    -1, EVAD0, EVAD0,    -1,    -1],#RETR0 Retrace steps
 [HOME0, HOME0, HOME0, HOME0, HOME0, HOME0, HOME0, HOME0, HOME0, HOME0,    -1,    -1, HOME0, HOME0,    -1,    -1],#HOME0 Return to start
 [RMOV0, ATCK0, RMOV0, ATCK0, EVAD0, EVAD0, EVAD0, EVAD0, EXPL0, ATCK0,    -1,    -1, EVAD0, EVAD0,    -1,    -1] #RMOV0 Random movement
]

