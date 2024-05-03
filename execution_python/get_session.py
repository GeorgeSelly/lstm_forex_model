import requests
import json
import os

base_path = r'C:\Users\georg\OneDrive\Documents\st0nks\forex'

env_obj = open(os.path.join(base_path, 'files', 'env.json'), 'r')
env = json.load(env_obj)
env_obj.close()

def get_session():
    response = requests.post(
        f"{env['auth']['base_url']}/session",
        headers={'Content-Type': 'application/json'},
        json = {
            "Password": env["auth"]["password"],
            "AppVersion":"1",
            "AppComments":"",
            "UserName": env["auth"]["username"],
            "AppKey": env["auth"]["app_key"]
        }
    )
    if (response.status_code == 200):
        return (json.loads(response.content.decode("utf-8"))['Session'])
    else:
        return None

session_id = get_session()
if session_id:
    tmp_file_obj = open(os.path.join(base_path, 'files', 'tmp.json'), 'r')
    tmp = json.load(tmp_file_obj)
    tmp_file_obj.close()

    print('SESSION ID SUCCESSFULLY RETRIEVED: ' + session_id)
    tmp["session_id"] = session_id
    tmp_file_obj = open(os.path.join(base_path, 'files', 'tmp.json'), 'w')
    tmp_file_obj.write(json.dumps(tmp))
    tmp_file_obj.close()