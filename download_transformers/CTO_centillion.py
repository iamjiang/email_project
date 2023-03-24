import os
import json
import random
import string
import requests
from datetime import datetime
from configparser import ConfigParser
import argparse
from tqdm import tqdm
 
BASE = "BASE"
PROJECT = "PROJECT"
TOKEN = "TOKEN"
PROXIES = {
    "https": "http://proxy.jpmchase.net:8443",
    "http": "http://proxy.jpmchase.net:8443"
}
#It works from linux not from vdi
def get_token():
    path=os.getenv("OMNI_USER_PKG_FRS_REQ").split("/")[1:-1]
    path.append("tokens.ini")
    path = "/" + "/".join(path)
    cfg = ConfigParser()
    cfg.read(path)
    access_token = cfg.get("TokensSection","token")
    return access_token
###
def download_artifact(api, model_name, model_version,):
    headers = {"Authorization": "Bearer {}".format(api[TOKEN]), "content-type": "application/json"}
    url_string = api[BASE] + "/v1/model-repo/model-artifacts/{}/{}/{}".format(api[PROJECT], model_name, model_version)
    response = requests.get(url=url_string, headers=headers, verify=True)
    print(response.url)
 
    status = response.status_code
    print("status code " , status)
    print("content = ", response.content)
    for item in json.loads(response.content):
        print("Downloading file {} ".format(item["fileName"]))
        download_url_string = api[BASE] + "/v1/model-repo/model-artifacts/download/{}/{}/{}/{}".format(api[PROJECT], model_name, model_version, item["fileName"])
        download_headers = {"Authorization": "Bearer {}".format(api[TOKEN])}
        print("start download time ********************",datetime.now().strftime("%Y%m%d-%H:%M:%S"))
 
        download_response = requests.get(url=download_url_string, headers=headers, verify=True, stream=True)
        downloaded_file = item["fileName"]#"/tmp/downloaded_"+datetime.now().strftime("%Y%m%d_%H%M%S")+item["fileName"]
        if download_response.status_code == 200:
            temp_file = open(downloaded_file, "wb")
            count = 0
            for chunk in tqdm(download_response.iter_content(chunk_size=4194304)):
                count = count + 1
                temp_file.write(chunk)
                #print ("chunk {} done".format(count))
            print("File {} downloaded as {} ".format(item["fileName"], downloaded_file))
        else:
               print("          ===============> download error....", download_response.status_code, download_response.content)
 
        print("End Download time ",datetime.now().strftime("%Y%m%d-%H:%M:%S"))
    return status
###
def main():
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        del os.environ['no_proxy']
    except:
        print("env already setup. No action taken.......")
    token = get_token()
    api = {
        BASE : "https://omni-modelrepo-api.prod.aws.jpmchase.net",
        PROJECT : "CTO Centillion",
        TOKEN : token
    }
    parser = argparse.ArgumentParser(description='get model_name and version')
    parser.add_argument('--model_name' , help='model_name')
    parser.add_argument('--model_version', help='model_version')
  
    args = parser.parse_args()
    download_artifact(api, args.model_name, args.model_version)
######
main()

