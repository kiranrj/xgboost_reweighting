```
conda env create -f ./scripts/environment.yml
conda activate xgb-influence
./make_figures.sh
# artifacts in ./out
```
```
conda activate xgb-influence
# either one of these:
pip install Jinja2
# or
conda install -c conda-forge jinja2
cd scripts
./make_figures.sh
```
