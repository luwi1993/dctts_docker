echo "PREPROCESS"
python dctts/prepo.py
echo "TRAIN TEXT2MEL"
python dctts/train_transfer.py 1
echo "TRAIN SSRN"
python dctts/train_transfer.py 2
echo "SYNTHESIZE"
python dctts/synthesize.py

