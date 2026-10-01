PY := .venv/bin/python
JUPYTER := .venv/bin/jupyter
NB := 01_data_acquisition_and_quality 02_portfolio_benchmarking_eui_carbon_water 03_monthly_bill_audit_and_anomalies \
      04_weather_normalisation_and_forecasting 05_ems_submeter_and_solar_pv 06_savings_opportunities_register 00_EXECUTIVE_REPORT

.PHONY: all setup data notebooks report readme test

all: data notebooks report readme

setup:
	python3.11 -m venv .venv && $(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps
	$(PY) -m ipykernel install --user --name wattwise --display-name "Python 3 (wattwise)"

data:
	$(PY) scripts/download_data.py

notebooks:
	@for n in $(NB); do echo "== $$n"; $(JUPYTER) nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.kernel_name=wattwise --ExecutePreprocessor.timeout=1800 notebooks/$$n.ipynb || exit 1; done

report:
	$(JUPYTER) nbconvert --to html --no-input notebooks/00_EXECUTIVE_REPORT.ipynb --output-dir reports --output WattWise_Report

readme:
	$(PY) scripts/render_readme.py

test:
	$(PY) -m pytest -q tests
