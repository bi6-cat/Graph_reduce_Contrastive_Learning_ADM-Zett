# Apk -> Prune -> Embedding 
```bash
    python -m core.cli.toolkit dataset core/config/config.yaml
```
# GNN
```bash
    python -m core.cli.toolkit training core/config/config.yaml
```

# Gnn embedidng
```bash
    python -m core.cli.toolkit gnn_embedding core/config/config.yaml
```

# Permission
```bash
    python -m core.cli.toolkit permission core/config/config.yaml
```

# Classify
```bash
    python -m core.cli.toolkit classify core/config/config.yaml
```

# Test config
```bash	
    pytest core/test/test_config.py -m all	
```