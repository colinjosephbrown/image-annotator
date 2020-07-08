import json


def db_diff(json_path_1, json_path_2, diff_path):
    not_in_db1 = []

    with open(json_path_1, 'r') as f1:
        db1 = json.load(f1)

    with open(json_path_2, 'r') as f2:
        db2 = json.load(f2)

    for img_fn, ann in db2.iteritems():
        if img_fn not in db1:
            not_in_db1.append(img_fn)

    print(not_in_db1)

    with open(diff_path, 'w') as fo:
        json.dump(not_in_db1, fo)


if __name__ == '__main__':
    json_path_1 = 'annotations.json'
    json_path_2 = 'unscaled_annots.bak.20190922_2037.json'
    diff_path = 'diff_annots.json'
    db_diff(json_path_2, json_path_1, diff_path)