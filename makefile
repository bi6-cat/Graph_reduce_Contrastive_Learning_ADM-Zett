dataset: 
	python -m core.cli.toolkit dataset core/config/config.yaml
ct_learning: 
	python -m core.cli.toolkit contastive_learning core/config/config.yaml
training: 
	python -m core.cli.toolkit training core/config/config.yaml
gnn_embedding:
	python -m core.cli.toolkit gnn_embedding core/config/config.yaml
permission:
	python -m core.cli.toolkit permission core/config/config.yaml
classify: 
	python -m core.cli.toolkit classify core/config/config.yaml
test_config:
	pytest core/test/test_config.py -m all	