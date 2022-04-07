import requests
import sys

if len(sys.argv) != 2:
    print("Usage: python discord_webhook.py MESSAGE")
    print()
    print("Missing argument MESSAGE")
    sys.exit(-1)

with open("webhook-url") as f:
    url = f.read()

#for all params, see https://discordapp.com/developers/docs/resources/webhook#execute-webhook
data = {}

#for all params, see https://discordapp.com/developers/docs/resources/channel#embed-object
data["embeds"] = [
    {
        "description" : f"{sys.argv[1]}",
        "title" : "Experiment"
    }
]

result = requests.post(url, json = data)

try:
    result.raise_for_status()
except requests.exceptions.HTTPError as err:
    print(err)
else:
    print("Payload delivered successfully, code {}.".format(result.status_code))

