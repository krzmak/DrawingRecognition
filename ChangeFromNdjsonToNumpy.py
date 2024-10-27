import json
from data_prep import NdjsonToNp    

with open('configuration.json', 'r') as config:
    location = json.load(config)

ndjson_data = location.get("raw ndjson data path")
raw_data = location.get("raw data path")

transform_to_np = NdjsonToNp(ndjson_data, raw_data) 

transform_to_np.ChangeFilesFormat()