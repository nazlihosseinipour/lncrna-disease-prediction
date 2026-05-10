# LDA-SABC
## Data
In this work，lncRNADisease is data1 and MNDR is data2.
## Environment
Install python3.7 for running this model. And these packages should be satisfied:

 - tensorFlow $\approx$ 1.14.0
 - keras $\approx$ 2.11.0
 - numpy $\approx$ 1.19.5
 - pandas $\approx$ 1.1.5
 - scikit-learn $\approx$ 0.24.2
## Usage
Extracting linear features for diseases and lncRNAs by SVD, to run:
```
python svd_feature/matrix_svd.py
```
Concatenation 
```
python data_handle.py
```
To run this model：
```
python main.py
```
