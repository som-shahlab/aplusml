import sys
import os
import re
import subprocess

################################################
# Goal: Run all 'test' scripts
################################################
test_files = sorted([f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and re.match("test\d+\.py", f)])
for t in test_files:
    subprocess.call([ "python3", t ])