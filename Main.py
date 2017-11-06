'''
Main program for running the PoE Bot
'''
from Bot import Bot
from BotDebugger import BotDebugger
import traceback
import time

bot = None

def BotLWTest():
    '''
    Perform a LW timing test
    '''
    b = GetBot()
    try:
        b.LWTest()
    except:
        traceback.print_exc()

def BotRun():
    '''
    Run the bot
    '''
    b = GetBot()
    try:
        b.Run()
    except:
        traceback.print_exc()
    b.Stop()
    
def BotRunDB():
    '''
    Run the bot with extra debugging windows/information
    '''
    b = GetBot()
    bd = BotDebugger(bot)
    bd.PlotLoop()
    try:
        b.Run()
    except:
        traceback.print_exc()
    bd.PlotStop()
    b.Stop()

def BotTSTest():
    '''
    Run the bot
    '''
    b = GetBot()
    try:
        b.TSTest()
    except:
        traceback.print_exc()
    
def GetBot():
    '''
    Only call constructor if object doesn't exist yet to save time.
    '''
    global bot
    if bot is None:
        print('Starting Bot...')
        bot = Bot("Path of Exile")
    return bot
    
#Add in all cells after given cell
if __name__ == "__main__":
    BotRun()