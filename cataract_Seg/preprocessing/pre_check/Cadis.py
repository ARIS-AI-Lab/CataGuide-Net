import json
import numpy as np

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data


def json_content_check(json_path):
    contents = load_json(json_path)
    for i in contents['annotations']:
        print(i)
        break


if __name__ == '__main__':
    json_file_path = r"C:\Users\Charl\Downloads\insegcat-2\cadis\training.json"
    json_content_check(json_file_path)