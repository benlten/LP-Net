FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 vim git tree g++ -y

RUN pip install powerline-shell tqdm pillow scikit-image tensorboard pytorch_lightning matplotlib pydantic opencv-python opencv-contrib-python pyyaml scikit-learn

RUN git clone https://github.com/shubhamkulkarni01/CSFiles ~/.setup_files && \
    ln -s -f ~/.setup_files/.bashrc ~/.bashrc && \
    ln -s -f ~/.setup_files/.bash_aliases ~/.bash_aliases && \
    ln -s -f ~/.setup_files/.gitconfig ~/.gitconfig && \
    ln -s -f ~/.setup_files/.print_time ~/.print_time && \
    ln -s -f ~/.setup_files/.condarc ~/.condarc && \
    ln -s -f ~/.setup_files/.vim ~/.vim && \
    ln -s -f ~/.setup_files/.vimrc ~/.vimrc && \
    mkdir ~/.ssh && \
    ln -s -f ~/.setup_files/config ~/.ssh/config && \
    mkdir -p ~/.config/powerline-shell && \
    ln -s -f ~/.setup_files/config.json ~/.config/powerline-shell/config.json && \
    ln -s -f ~/.setup_files/kubectl.py ~/.config/powerline-shell/kubectl.py

CMD "/bin/bash"
