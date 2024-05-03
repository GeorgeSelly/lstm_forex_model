import runpy
import schedule
import datetime as dt
import time
import os

base_path = r'C:\Users\georg\OneDrive\Documents\st0nks\forex'

# Functions

def get_session():
    runpy.run_path(path_name=os.path.join(base_path, 'execution_python', 'get_session.py'))

def parse_stream_files():
    print(dt.datetime.now())
    runpy.run_path(path_name=os.path.join(base_path, 'execution_python', 'parse_stream_files.py'))


def test_func():
    print('Hello world!')

# Set up schedule object


schedule.every(15).minutes.do(get_session)
#schedule.every(5).seconds.do(test_func)

# main
get_session()
while True:
    schedule.run_pending()
    time.sleep(1)

