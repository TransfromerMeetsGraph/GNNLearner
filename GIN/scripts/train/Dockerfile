FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y vim

# Used for OGB
RUN conda install -y -c dglteam dgl-cuda10.1

RUN conda install -y numpy scipy matplotlib ipython
RUN conda install -y scikit-learn
RUN conda install -y tqdm

# Used for RDKit
RUN apt-get install -y libxrender1 libxext6
RUN conda install -y rdkit>=2019.03.1 -c rdkit

RUN conda install -y tensorboard==2.5.0

RUN python -m pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html \
    && python -m pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html \
    && python -m pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html \
    && python -m pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html \
    && python -m pip install torch-geometric
