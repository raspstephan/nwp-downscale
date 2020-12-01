from ecmwfapi import ECMWFDataServer
from fire import Fire

def main(start_date, stop_date, fn):
    server = ECMWFDataServer()
    server.retrieve({
        "class": "yp",
        "dataset": "yopp",
        "date": f"{start_date}/to/{stop_date}",
        "expver": "1",
        "levtype": "sfc",
        "param": "228.128",
        "step": "0/3/6/9/12/15/18/21/24/27/30/33/36/39/42/45/48/51/54/57/60/63/66/69/72",
        "stream": "enfo",
        "time": "00:00:00/12:00:00",
        "type": "cf",
        "target": fn,
        "format": "netcdf"
    })
    
if __name__ == '__main__':
    Fire(main)