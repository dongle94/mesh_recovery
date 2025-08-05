#!/bin/bash

# Script that fetches all necessary data for training and eval

# Model constants etc.
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
# Initial fits to start training
# wget http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz && tar -xvf static_fits.tar.gz --directory data && rm -r static_fits.tar.gz
# List of preprocessed .npz files for each dataset
# wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
# Pretrained checkpoint
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt
mv model_checkpoint.pt hmr.pt
# Uncomment the following line to download the initial fits for training 
# wget http://visiondata.cis.upenn.edu/spin/spin_fits.tar.gz && tar -xvf spin_fits.tar.gz && rm -r spin_fits.tar.gz


# GMM prior from vchoutas/smplify0x
# wget https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py -O smplify/prior.py
