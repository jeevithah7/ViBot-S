import urllib.request
import json
with urllib.request.urlopen('http://localhost:5000/stream/events') as response:
    for line in response:
        line = line.decode('utf-8').strip()
        if line.startswith('data: '):
            data = json.loads(line[6:])
            print("Received JSON keys:", list(data.keys()))
            if 'path' in data:
                print("Path length:", len(data['path']))
            if 'grid' in data:
                print("Grid rows:", len(data['grid']), "cols:", len(data['grid'][0]))
            break
