# Real-world deployment of remote sleep monitoring technologies reveals distinct patterns associated with cognitive decline

This is the repository associated with the paper "Real-world deployment of remote sleep monitoring technologies reveals distinct patterns associated with cognitive decline".

## Files

Here, we provide a description of the files made available in this repository.

### The Models

To promote the sharing of resources, we provide the pre-trained age estimation model ('age_estimator.json') and final risk prediction model ('risk_predictor.pkl') described in the paper. 

### Scripts

This folder contains all associated code (including scripts for data pre-processing and stratification).

### Data

The data presented in this study came from three separate sources. The Withings dataset was provided under a data-sharing agreement for research with Imperial College London and cannot be shared. The Minder dataset is available from the corresponding authors upon reasonable request. The Resilient dataset has been made publicly available and can be found on Zenodo at: https://zenodo.org/records/16755408.

### Experiments

Code for experiments and figures presented in this study will be made available by the corresponding author upon reasonable request.

## Set-up

For this, you will need to have conda installed (find more information here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Create the environment from the environment.yml file:

```python
conda env create -f environment.yml
```

Activate the environment:

```python
conda activate sleep-age
```

Verify that the environment was installed correctly:

```python
conda env list
```

## Running the Model

To generate outputs on your own data, you can run the python scripts 'age_estimation.ipynb' and 'risk_prediction.ipynb'.



## Contact

This code in maintained by Nan Fletcher-Lloyd.
