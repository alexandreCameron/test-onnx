# test-onnx on sklearn

## Purpose

Test onnx-export of sklearn transform function.

This repo is not meant to create efficient code.

## Configure virtual environment

* [ ] Run:

```bash
python3 -m venv venv3_test-onnx
```

* [ ] Activate virtual environment:

```bash
source venv3_test-onnx/bin/activate
```

* [ ] Install packages:

```bash
pip3 install -r requirements.txt 
```

NB: this `requirements.txt` has more packages than necessary to test onnx.
These packages are standard ML packages.

* [ ] Add kernel to jupyter notebook:

```bash
python3 -m ipykernel install --user --name venv3_test-onnx --display-name "venv3_test-onnx"
```

* [ ] Install graphviz to view representation of graphs:

```bash
apt-get install graphviz
```
