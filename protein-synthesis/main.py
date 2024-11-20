from protein_synthesis_scenarios import *

def main():
    functions = [
        transcription_only,
        simple_transcription_and_simple_translation,
        simple_expression_mrna_copy_number_around_2,
        simple_expression_mrna_copy_number_around_2_high_transcription_low_translation_rate,
        self_regulation_transcription_and_simple_translation_h_and_k_trials,
        keep_h_at_2_and_increasing_k,
        self_regulating_transcription_and_simple_translation,
        simple_transcription_and_unmatured_protein_translation,
        self_regulating_and_unmatured_protein_translation
    ]

    print("Choose a function to run:")
    for i, func in enumerate(functions):
        print(f"{i + 1}. {func.__name__}")

    choice = int(input("Enter the number of the function to run: ")) - 1

    if 0 <= choice < len(functions):
        functions[choice]()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()