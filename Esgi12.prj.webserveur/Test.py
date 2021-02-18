from flask import Flask
import urllib.request, json

version = "1"

app = Flask(__name__)
@app.route('/')
def recupe_json():
    with urllib.request.urlopen("https://esgi-devops12tp3-app.azurewebsites.net/weatherforecast") as url:
        data = json.loads(url.read().decode())
        a=[]
        text = ''
        for rowJson in data:
            row = json.dumps(rowJson)
            row = row.replace("{\"date\"", "Date ")
            row = row.replace("\"temperatureC\"", "Temperature °C ")
            row = row.replace("\"temperatureF\"", "Temperature °F ")
            row = row.replace("\"summary\"", "Summary ")
            row = row.replace("}","")
            a.append((row))
    for element in a:
        text += ('<li>'+str(element)+'</li>')
    return ('<h1>Bienvenue </h1>'+'<br>'+ '<table><thead><tr><th colspan="2">Météo</th></tr></thead><tbody><tr><td>'+text+'</tbody></tr></td></table><br><text>Version :'+version+'</texte>')



