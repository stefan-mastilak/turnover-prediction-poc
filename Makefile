# Makefile

help:
	@echo  -------------------------------------------------------------------------------------------
	@echo  make notebook - command to open notebooks folder in jupyter lab
	@echo  make notebook_save - command to save .ipynb files in notebooks folder
	@echo  make notebook_save_py - command to save .ipynb files as .py files in notebooks folder
	@echo  -------------------------------------------------------------------------------------------

notebook:
	jupyter lab notebooks/ > ./jupyterlab.log 2>&1 &

notebook_save:
	jupytext --output notebooks/cls_models.py notebooks/cls_models.ipynb
	jupytext --output notebooks/data_analysis.py notebooks/data_analysis.ipynb
	jupytext --output notebooks/data_distributions.py notebooks/data_distributions.ipynb

notebook_load:
	jupytext --to notebooks/cls_models.py
	jupytext --to notebooks/data_analysis.py
	jupytext --to notebooks/data_distributions.py
