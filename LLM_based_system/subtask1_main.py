import configs
from user_aware_test import run_user_aware_prompt_static
from user_agnostic_test import run_user_agnostic_prompt
from data_checks import check_missing_ids, check_naming_consistency, merge_json_files_with_buckets, merge_json_files_without_buckets
from help_funtions import generate_full_csv_for_submission


if __name__ == "__main__":
    # PARAMS
    uaw_file_name = r"results\uaw_results"
    uag_file_name = r"results\uag_results"
    merged_uag_file_name = r"results\merged_uag_results.json"
    all_file_name = r"results\all_results.json"
    submission_file_name = r"results\subtask1_submission.json"

    uaw_file = run_user_aware_prompt_static(
        save_to_file_name=uaw_file_name,
        openai=True,
        model_name=configs.model_openai_gpt_5,
        prompt_type="emotion",
        train_len=20,
        from_b=0,
        to_b=200
    )
    uag_file = run_user_agnostic_prompt(
        prompt=configs.prompt_15shot,
        openai=True,
        model_name=configs.model_openai_gpt_5_1,
        shuffled=True,
        test_data=True,
        save_to_file_name=uag_file_name,
        num_of_buckets=125,
        from_bucket=0,
        to_bucket=125
    )

    merged_uag = merge_json_files_with_buckets(merged_uag_file_name, [uag_file])
    merged_all = merge_json_files_without_buckets(all_file_name, [merged_uag, uaw_file])

    check_missing_ids(merged_all, test_data=True, full_data=True)
    check_naming_consistency(merged_all, full_data=True)

    generate_full_csv_for_submission(in_file_name=all_file_name, out_file_name=submission_file_name, test_data=True, zip_output=False)




