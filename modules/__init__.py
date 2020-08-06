
from os.path import dirname, join, abspath
import sys
DIR = join(dirname(str(abspath(__file__))), 'gsplines/')
sys.path.append(DIR)
import gsplines
