#!/bin/bash
if [ -e ./kaggle.json ]
then
    cp ./kaggle.json /root/.kaggle/kaggle.json
    chmod 600 /root/.kaggle/kaggle.json
    echo "Downloading dataset!"
    kaggle datasets download -d diyer22/retail-product-checkout-dataset
    echo "Unzipping dataset..."
    unzip -q 'retail-product-checkout-dataset.zip'
else
    if [ -e /root/.kaggle/kaggle.json ]
    then
        cp kaggle.json /root/.kaggle/kaggle.json
        chmod 600 /root/.kaggle/kaggle.json
        echo "Downloading dataset!"
        kaggle datasets download -d diyer22/retail-product-checkout-dataset
        echo "Unzipping dataset..."
        unzip -q 'retail-product-checkout-dataset.zip'
    else    
        echo "Cannot locate kaggle.json to download the dataset.
        Kaggle needs kaggle.json file to be in the root directory to download a dataset to colab. 
        To generate that json head to your account page and Account tab on that page: https://www.kaggle.com/settings/account. Then, click "Create New Token" button to download your own kaggle.json file. This is specific for user account. 
        Upload the json file to the runtime and run below cell to move the file to the required folder."
    fi
fi