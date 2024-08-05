# mat_inv_design
## Usage:
### 1. Training 
#### 1. Prepare the data
Extract perov5.zip data file under data folder
#### 2. Run train.py and then postprocess.py
    python train.py data/perov5
    python postprocess.py data/perov5
### 2. Encoding 
#### Use the date as unique model ID
    python autoencoder.py --date yyyymmddhhmm --task encoding data/perov5
### 3. DDPM training and sampling
#### Ues vectors encoded above as input data
    accelerate launch ddpm.py --date yyyymmddhhmm 
### 4. Decoding
#### Decode encoded or sampled vectors into CIF files, with the predicted properties saved in .txt file
    python  autoencoder.py --date yyyymmddhhmm --task decoding --rot-id yyyymmddhhmm --trans-id yyyymmddhhmm results/sampled/yyyymmddhhmm
### 5. Checking decoded structures with [visualization.ipynb](visualization.ipynb)