from ecmwfapi import ECMWFDataServer
from fire import Fire

def main(start_date, stop_date, fn, ensemble=False):
    server = ECMWFDataServer()
    request = {
        "class": "ti",
        "dataset": "tigge",
        "date": f"{start_date}/to/{stop_date}",
        "expver": "prod",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "228228",
        "step": "0/6/12/18/24/30/36/42/48/54/60/66/72",
        "time": "00:00:00/12:00:00",
        "type": "cf",
        "target": fn,
        # "format": "netcdf"
    }
    if ensemble:
        request['number'] = "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50"
        request['type'] = 'pf'
    server.retrieve(request)
    
if __name__ == '__main__':
    Fire(main)