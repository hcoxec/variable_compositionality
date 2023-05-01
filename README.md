# Compositionality With Variation Reliably Emerges Between Neural Networks

Code for ICLR 2023 Paper [Compositionality With Variation Reliably Emerges Between Neural Networks](https://openreview.net/pdf?id=-Yzz6vlX7V-)

(I'm presenting this at ICLR Wednesday - come say hi)

### Setup
Create a new virtual environment and run:

```bash scripts/setup.sh```

to install required packages. Aternately install the packages in **scripts/requirements.txt** however you like.

### Running Jobs
Each run needs a config file, the config used for the experiments in the paper can be found under **configs/iclr_2023.jsonnet**

Each run has 3 parts, **preprocess, train, eval.**
- **preprocess** reads input output pairs from a tsv in the **/data** directory
- **train** runs the model
- **eval** runs post-hoc analyses on the checkpoints saved during training

To run each of these steps navigate to the **/code** directory and run:

```python main.py preprocess configs/iclr_2023.py```

then:

```python main.py train configs/iclr_2023.py```

then:

```python main.py eval configs/iclr_2023.py```

Included under the scripts directory is a file **run.sh** which runs all of these steps by just calling:

```bash run.sh```

 Also included is **run_array.sh** which runs the model for 10 initializations and provides an example of how to add commandline arguments, for run dependent parameters 0 like random seed.

#### Attribution

This code is made available under the MIT software license, and uses portions of the code from the Facebook Research [EGG](https://github.com/facebookresearch/EGG) Repo made available under the same license. Notes are made in the code where objects are based on EGG.