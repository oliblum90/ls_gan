import os
import sys
import shutil

def init_project(name):
    
    try:
        os.mkdir(name)
    except:
        decission = input('Do you want to delete existing project? (1 = yes, 2 = no)')
        if decission == 1:
            shutil.rmtree(name)
            os.mkdir(name)
        else:
            print "end programm"
            sys.exit()