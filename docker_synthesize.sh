sudo docker run -it -v "/home/lwidowski/Documents/sound/dctts_docker/logdir/:/dctts/logdir/:rw" -v "/home/lwidowski/Documents/sound/dctts_docker/results/:/dctts/samples/:rw" -v "/home/lwidowski/Documents/sound/dctts_docker/text_input:/dctts/text_input:rw" dctts python /dctts/synthesize.py