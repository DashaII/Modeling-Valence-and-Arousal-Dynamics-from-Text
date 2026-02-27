Instructions:
1. Install the required packages: pip install openai numpy pandas
2. Add your OpenAI API key in confgs.openai_key
3. Adjust the raw_train_data and raw_test_data paths if needed (all required data is provided in the data folder).
4. Run subtask1_main.py
5. The results will be saved to: results\subtask1_submission.json


notes:
 - The results may differ from the submitted ones, as we used the default OpenAI temperature (1.0, according to the official documentation).
 - train_data_emotion.csv contains training data enriched with emotion labels
 - test_subtask1_enriched.csv contains both test and training data with additional columns:
   - valence, arousal: available for training examples only 
   - train_data flag: TRUE for training examples, FALSE for test examples 
   - emotion: emotion label for training examples, UNKNOWN for test examples