# Install env with conda:
- Setup miniconda: 
```bash
    # step 1
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

    # Step 2
    source ~/miniconda3/bin/activate

    # Step 3
    conda init --all
```
- Init my env:
```bash
    conda env create --file=environment.yaml
```