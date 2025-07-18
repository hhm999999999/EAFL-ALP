#!/usr/bin/env bash
export PYTHONPATH=.
python -m EAFL_ALP.client.cli run --num-clients 5 "$@"