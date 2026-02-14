conda create -yn QCQC python=3.9
# conda run -n QCQC pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
conda run -n QCQC conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda run -n QCQC pip install setuptools
conda run -n QCQC pip install --no-build-isolation git+https://github.com/openai/CLIP.git
conda run -n QCQC pip install sentencepiece transformers lmdb psutil
conda run -n QCQC pip install open_clip_torch
conda run -n QCQC pip install wandb
conda run -n QCQC pip install scikit-learn
conda run -n QCQC pip install pandas
conda run -n QCQC pip install ipdb
conda run -n QCQC pip install datasets
conda run -n QCQC pip install accelerate -U
conda run -n QCQC conda install -y pytorch::faiss-gpu=1.8.0
conda run -n QCQC pip install pycocotools