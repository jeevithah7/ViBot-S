import os

path = 'c:/Users/laya/Desktop/PE/ViBot-S/dashboard/app.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(path, 'w', encoding='utf-8') as f:
    for line in lines:
        if line.startswith('DASHBOARD_HTML = '):
            break
        f.write(line)
print("Truncated app.py successfully.")
