from training_subtask2a import run_inference

# ========================================
# MAIN AROUSAL
# ========================================

if __name__ == "__main__":
    # Configuration
    PREDICT_TARGET = 'arousal'  # 'valence', 'arousal', or 'both'
    WINDOW_SIZE = 1  # Start with 1 based on your findings
    USE_TEXT = False  # Set to False to test no-text baseline
    USER_EMB_DIM = 4  # Smaller user embedding dimension
    USE_WORDS = False  # Set to False to test no-words baseline
    MODEL_NAME = 'FacebookAI/roberta-base'

    TRAIN_PATH = "./train_data_padded.csv"
    TEST_PATH = "./test_data_padded.csv"
    TRAIN_DATASET_FLAG = False

    # Inference
    print("\n" + "=" * 50)
    print("INFERENCE")
    print("=" * 50)

    if USE_TEXT:
        if USE_WORDS:
            output_suffix = 'withwords'
        else:
            output_suffix = 'withtext'
    else:
        output_suffix = 'notext'

    model_for_file = MODEL_NAME.replace('/', '_')

    predictions = run_inference(
        test_path=TEST_PATH,
        train_dataset=TRAIN_DATASET_FLAG,
        checkpoint_path="model_for_arousal.pth",
        batch_size=32,
        output_path=f'predictions_window_{WINDOW_SIZE}_{model_for_file}_user{USER_EMB_DIM}_{PREDICT_TARGET}_{output_suffix}_all.csv',
        model_name='FacebookAI/roberta-base'
    )