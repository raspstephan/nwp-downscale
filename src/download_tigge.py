from ecmwfapi import ECMWFDataServer
from fire import Fire

def main(start_date, stop_date, fn):
    server = ECMWFDataServer()
    server.retrieve({
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
    })
    
if __name__ == '__main__':
    Fire(main)