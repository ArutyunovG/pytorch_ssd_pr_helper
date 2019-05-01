#!/bin/bash

if [ ! -f "ConvertedModels.tar.gz" ]; then
	wget https://www.dropbox.com/s/pdcji6rhw5xf6hk/ConvertedModels.tar.gz
	tar -zxvf ConvertedModels.tar.gz
fi

python run_models.py

