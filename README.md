## FALCON: Fast Image Haze Removal Leveraging Continuous Density Mask
## Installation

- Python 3.9
- PyTorch 1.12.1

	```
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12 1 —extra-index-url https://download.pytorch.org/whl/cu116
	pip install -r requirements.txt
    ```
## Preparing Dataset
### Dense Haze
https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/

### NH Haze1
https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/

### NH Haze2
https://competitions.codalab.org/competitions/28032

## Training
```
cd src
bash scripts/train.sh
```

## Testing
- downloads checkpoints at this [LINK](https://drive.google.com/file/d/1sJ3sapQxFzFMWw-NmJQsmWJPwDzq4xjj/view?usp=sharing)
```
cd src
mkdir ckpts
```

- Upload to the checkpoint at correct path(`src/ckpts`).

```
bash scripts/test.sh
```

## Acknowledgements
Part is our code is built upon modules of [LaMa](https://github.com/advimman/lama). Thanks for their impressive work!
