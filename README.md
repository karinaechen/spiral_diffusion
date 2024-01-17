# Diffusion Model to Reverse Noisy Swiss Roll Data

This project explores diffusion models using a simple 2D dataset, investigates how points move during the diffusion process, and examines the effects of various hyperparameters on model success.


## Running the Project

- To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`
- To change the parameters, edit the `config.json` file

### Building the project stages using `run.py`

- To generate the data, run python `run.py data` from the project root dir
- To train the model (which includes generating the data), run `python run.py all` from the project root dir
  - This creates a model based on the parameters in `config.json` and saves it in the `models` directory
- To test the whole process using a unit test, run `python run.py test` from the project root dir

## Reference

https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_03_waveform.ipynb

