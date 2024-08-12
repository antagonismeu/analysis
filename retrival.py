import requests
import json
import csv


headers = {'Content-type': 'application/json'}


series_ids = [
    'CUUR0000SA0', 'SUUR0000SA0', 'CUUR0000SA1', 'SUUR0000SA1',
    'CUUR0000SA2', 'SUUR0000SA2', 'CUUR0000SA3', 'SUUR0000SA3',
    'CUUR0000SA4', 'SUUR0000SA4', 'CUUR0000SA5', 'SUUR0000SA5',
    'CUUR0000SA6', 'SUUR0000SA6', 'CUUR0000SA7', 'SUUR0000SA7',
    'CUUR0000SA8', 'SUUR0000SA8', 'CUUR0000SA9', 'SUUR0000SA9',
    'CUUR0000SAA', 'SUUR0000SAA', 'CUUR0000SAB', 'SUUR0000SAB'
]


end_year = "2023"


start_year = str(int(end_year) - 9)


with open('data.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    
    csv_writer.writerow(["series id", "year", "period", "value", "footnotes"])
    
    
    for i in range(7):

        for series_id in series_ids :

            data = json.dumps({
                "seriesid": [series_id],
                "startyear": start_year,
                "endyear": end_year
            })
        
            
            response = requests.post('https://api.bls.gov/publicAPI/v1/timeseries/data/', data=data, headers=headers)
            json_data = json.loads(response.text)
            
            
            for series in json_data['Results']['series']:
                series_id = series['seriesID']
                for item in series['data']:
                    year = item['year']
                    period = item['period']
                    value = item['value']
                    footnotes = ""
                    for footnote in item['footnotes']:
                        if footnote:
                            footnotes += footnote['text'] + ','
                    
                    
                    if 'M01' <= period <= 'M12':
                        csv_writer.writerow([series_id, year, period, value, footnotes.rstrip(',')])

print("Data has been written to data.csv")