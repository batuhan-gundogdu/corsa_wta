# corsa_wta
zerospeech 2020 submission for bogazici university

To train the CoRSA model :

train.py --data_dir=/path/to/segmented/utterances

This will save the model into temp.pth (name of which you can set with the --exp_name argument)

With the model, predict and compress the test uterances with the following :

predict.py --test_data_dir=/path/to/test/utterances

The resulting compressed embeddings are saved in the /Results folder
