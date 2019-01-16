## Requirements

- Python 3.6 (It will not work with python 3.7)


## Run

Run it in background

    nohup python3.6  -u avito_code.py /home/semantic/mpd.v1/data/ /home/semantic/mpd.v1/challenge_set.json > log_avito.log &
    tail -f log_avito.log

Check logs

    tail -f log_avito.log


