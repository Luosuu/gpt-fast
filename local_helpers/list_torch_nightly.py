# refer: https://discuss.pytorch.org/t/install-particular-pytorch-nightly/186853/7
# example wheel for this platform: https://download.pytorch.org/whl/nightly/cu124/torch-2.5.0.dev20240901%2Bcu124-cp310-cp310-linux_x86_64.whl 
import datetime
from datetime import date
import requests

seed_template = 'https://download.pytorch.org/whl/nightly/cu124/torch-2.{}.0.dev{}%2Bcu124-cp310-cp310-linux_x86_64.whl'
seed_date = date(2024, 9, 1)
seed_version = 5
last_success_date = seed_date
tolerance = datetime.timedelta(days=3)
current_date = seed_date
current_version = seed_version
current_tolerance = tolerance

while True:
    # date should be YYYYMMDD
    while current_date - last_success_date < current_tolerance:
        current_url = seed_template.format(current_version, current_date.isoformat().replace('-', ''))
        current_date = current_date - datetime.timedelta(days=1)
        if requests.head(current_url).status_code == 200:
            print(current_url)
            last_success_date = current_date
            break
        else:
            current_tolerance = current_tolerance - datetime.timedelta(days=1)
            if current_tolerance < datetime.timedelta(days=0):
                current_tolerance = tolerance
                current_version = current_version - 1
                if current_version < 0:
                    exit(0)