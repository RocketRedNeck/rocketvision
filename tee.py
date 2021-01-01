# Standard imports
import colorama
from colorama import Fore, Back, Style
import datetime
import os
import re
import sys
import threading

# 3rd party imports

# Local imports
from .freeze_it import freeze_it

colorama.init(convert=True)

'''
  Create a simple redirection to Tee output to both console and a file
  This works for ALL console output intended for stdout (even by internal
  processes being called that display to the console from inside the python
  shell.
  This pattern probably shows up in 1000's of places on the interwebz
  May want this added to our own tool set so we don't need to redefine
  it repeatedly
  Can also consider modification to pass in a more general object with a write operator
  to remove the file context and make it any other type of transport mech (e.g., could
  be a socket or some other interface)... but for now it is a filename
'''
@freeze_it
class Tee(object):
    def __init__(self, filename = "Tee.txt", fileoption = "w+", append = False, timestamp = False):
        self.lock = threading.Lock()
        self._timestamp = timestamp
        self.ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
        if isinstance(sys.stdout, Tee):
            # A Tee already exists, we can either append
            # or reset to only simple 2-way behavior
            if not append:
                # If not appending, then roll back one level
                # to get the previous console object
                sys.stdout.file.close()
                sys.stdout = sys.stdout.console
        self.console = sys.stdout
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.file    = open(filename, fileoption)
        except Exception as e:
            self.file    = None
            print(filename + " : FAILED TO OPEN : " + e)

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if not isinstance(value,bool):
            raise TypeError('value must be bool type')

        self._timestamp = value

    def escape_ansi(self, line):
        ''' remove ansi escape sequences when writing to files
        '''
        return self.ansi_escape.sub('', line)

    def write(self, message):
        nowstr = str(datetime.datetime.now())
        self.lock.acquire()

        # Use <y> </y> as delimiter for yellow highlight
        if '<y>' in message:
            message = message.replace('<y>',Fore.BLACK + Back.YELLOW)
            message = message.replace('</y>',Style.RESET_ALL)

        # Use <m> </m> as delimiter for magenta highlight
        if '<m>' in message:
            message = message.replace('<m>',Fore.BLACK + Back.MAGENTA)
            message = message.replace('</m>',Style.RESET_ALL)

        # Use <c> </c> as delimiter for cyan highlight
        if '<c>' in message:
            message = message.replace('<c>',Fore.BLACK + Back.CYAN)
            message = message.replace('</c>',Style.RESET_ALL)

        # Use <g> </g> as delimiter for green highlight
        if '<g>' in message:
            message = message.replace('<g>',Fore.BLACK + Back.GREEN)
            message = message.replace('</g>',Style.RESET_ALL)

        # Use <b> </b> as delimiter for blue highlight
        if '<b>' in message:
            message = message.replace('<b>',Fore.WHITE + Back.BLUE)
            message = message.replace('</b>',Style.RESET_ALL)

        # Use <r> </r> as delimiter for red highlight
        if '<r>' in message:
            message = message.replace('<r>',Fore.WHITE + Back.RED)
            message = message.replace('</r>',Style.RESET_ALL)

        # Use <w> </w> as delimiter for white highlight
        if '<w>' in message:
            message = message.replace('<w>',Fore.BLACK + Back.WHITE)
            message = message.replace('</w>',Style.RESET_ALL)

        upstr = message.upper()

        if '[FAIL]' in upstr or '[ERROR]' in upstr:
            message = Fore.RED + Style.BRIGHT + message + Style.RESET_ALL
        elif '[WARNING]' in upstr or 'DEPRECATIONWARNING' in upstr:
            message = Fore.YELLOW + Style.BRIGHT + message + Style.RESET_ALL
        elif '[PASS]' in upstr or '[SUCCESS]' in upstr:
            message = Fore.GREEN + Style.BRIGHT + message + Style.RESET_ALL
        elif '[INFO]' in upstr :
            message = Fore.CYAN + Style.BRIGHT + message + Style.RESET_ALL
        elif '# ' in upstr:
            message = Fore.GREEN + Style.DIM + message + Style.RESET_ALL
        
        if self.timestamp and len(message) > 3:
            message = f'[{nowstr}] {message}'
            
        self.console.write(message)

        try:
            self.console.flush()   # Keep user up to date, without this there can be lag from some shells
        except:
            pass # A redirect does not support flush
            
        if (self.file != None):
            # Any escape sequences from caller or above are striped
            # when sending to file
            message = self.escape_ansi(message)
            if len(message) > 0:
                self.file.write(message)
                self.file.flush()
        
        self.lock.release()

    def flush(self):
        # For compatibility pass the console flush down until the real console is invoked
        # This is the case if the user has forked (appended) the Tee
        # In the future this will be important because the Tee may go to
        # console, file, and say something like an Ethernet destination
        if isinstance(self.console, Tee):
            self.console.console.flush()

def simple_log(modulefile, timestamp = False, datastorage_path = '.'):
    # Capture this module path, and base file name without the extension
    rootdir = os.path.dirname(os.path.abspath(modulefile))
    thisfilename = os.path.basename(modulefile)
    thismodname = os.path.splitext(thisfilename)[0]

    if not os.path.exists(datastorage_path):
        os.mkdir(datastorage_path)    # Raises exception for bad paths

    # Redirect output (Tee) to console and logfile
    logfile = os.path.join(datastorage_path,thismodname + ".txt")
    sys.stdout = Tee(logfile, timestamp = timestamp)
    sys.stderr = sys.stdout

    print(Style.RESET_ALL)

    return thismodname
