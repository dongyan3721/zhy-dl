

import json
t = []
with open('../c2e/cmn.txt', 'r') as f:
    while True:
        line = f.readline()
        x = line.split('\t')
        t.append(x[:2])
        if not line:
            break

to_remove = []

for i in range(len(t)):
    if len(t[i]) < 2:
        to_remove.append(i)

t.pop(*to_remove)

with open('../c2e/cmn_json.json', 'w') as f:
    json.dump(t, f)

with open('../c2e/t.cn', 'w', encoding='utf-8') as cn:
    cn.write('\n'.join([x[1] for x in t]))


with open('../c2e/t.en', 'w', encoding='utf-8') as en:
    en.write('\n'.join([x[0] for x in t]))

