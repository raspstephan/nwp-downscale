from ecmwfapi import ECMWFDataServer
from fire import Fire
import numpy as np

def main(start_date, stop_date, fn, ensemble=False, members=50, lead_time=72):
    server = ECMWFDataServer()
    request = {
        "class": "ti",
        "dataset": "tigge",
        "date": f"{start_date}/to/{stop_date}",
        "expver": "prod",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "228228",
        "step": '/'.join(np.arange(0, lead_time+6, 6).astype(str)),
        "time": "00:00:00/12:00:00",
        "type": "cf",
        "target": fn,
        # "format": "netcdf"
    }
    if ensemble:
        request['number'] = '/'.join(np.arange(1, members+2).astype(str))
        request['type'] = 'pf'
    server.retrieve(request)
    
if __name__ == '__main__':
    Fire(main)