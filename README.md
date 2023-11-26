# Local-Global methods for Generalised Solar Irradiance Forecasting

This repository contains the implementations for the methods and supplementary material for paper "Local-Global methods for Generalised Solar Irradiance Forecasting"


## Paper
```
@article{,
  title = {Local-Global methods for Generalised Solar Irradiance Forecasting},
  author = {Cargan, T and Landa-Silva, D and Triguero, I},
  year = {2023},
}
```

### 
All the model code can be found in the folder `/expers`:

 | Model        | File                   |
 |--------------|------------------------| 
 | CNN          | `simple_cnn.py`        |
 | DNN          | `simple_dnn.py`        |
 | LSTM         | `simple_lstm.py`       |
 | Transformer  | `simple_transfomer.py` |


## Results
The raw results from the experiments outlined in the paper can be found the `/results` folder.

# Running Experiments
## Installing
The easiest way is to clone the repo and run flowing commands to install everything needed to run the experiments:
```shell
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install chemise @ git+https://github.com/TimCargan/chemise.git@master
pip install hemera @ git+https://github.com/TimCargan/hemera.git@master
pip install -e .
```

Then to run some models
```shell
python ./expers/simple_conv.py simple_conv.py --batch_size 64 --learning_rate 3e-4 --num_epochs 20
```

### Data
The satellite image data can be downloaded from EUMETSAT using our downloader scripts: [EUMETSAT-downloader](https://github.com/TimCargan/eumetsat-downloader)
All irradiance and weather data is available on request, we are working to have permission to upload it to hugging face.

