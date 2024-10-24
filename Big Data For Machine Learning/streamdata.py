import json
import requests
r = requests.get('http://stream.meetup.com/2/open_events', stream=True)
for line in r.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        print(json.loads(decoded_line))
