import os
import sys

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, PROJECT_HOME)
from app import app as application
