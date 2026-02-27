Instructions:
1. Install the required packages: pip install pandas torch transformers tqdm
2. To generate arousal predictions: run arousal_inference.py
3. To generate valence predictions: python valence_inference.py 
4. Each script loads the corresponding pretrained model - model_for_arousal.pth / model_for_valence.pth - and produces prediction files in the expected submission format.

Notes:
- Results may differ slightly from the originally submitted ones due to stochastic effects during model training.
- arousal_inference.py and valence_inference.py contain the respective training parameters used for each model. These parameters should be used when running training_subtask2a.py to reproduce the provided model weights. 
- train_data_padded.csv contains training data enriched with padded historical context for users with a small number of entries.
- test_data_padded.csv is identical to train_data_padded.csv except that it includes one additional entry per user, which is the instance to be predicted.