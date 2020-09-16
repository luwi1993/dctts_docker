#from nutzio/librosa-env
from tensorflow/tensorflow

run apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

run apt-get install libsndfile1 --yes
run apt-get install vim --yes

run pip install tensorflow==1.4
run pip install --upgrade pip
run pip install update pip
run pip install numpy
run pip install librosa
#run pip install tensorflow-1.3.0
run pip install tqdm 
run pip install matplotlib
run pip install scipy

run mkdir /dctts
run git clone https://github.com/Kyubyong/dc_tts.git /dctts
copy cfg/hyperparams.py /dctts/hyperparams.py
#copy dctts/ /dctts/
run mkdir /dctts/text_input
run mv /dctts/harvard_sentences.txt /dctts/text_input/harvard_sentences.txt
run mkdir /dctts/logdir/
run mkdir /dctts/samples/
