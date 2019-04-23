#!/bin/bash

if [ ! -f "ConvertedModels.tar.gz" ]; then
	wget https://www.dropbox.com/s/l9cudg1n54sswfo/ConvertedModels.tar.gz
	tar -zxvf ConvertedModels.tar.gz
fi

python run_models.py

