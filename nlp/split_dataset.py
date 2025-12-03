
import random

def split_data(en_file_path, cn_file_path, test_ratio=0.2):
    """
    Reads parallel corpus files, shuffles them, and splits them into
    training and testing sets. The output is printed to the console.

    Args:
        en_file_path (str): Path to the English sentences file.
        cn_file_path (str): Path to the Chinese sentences file.
        test_ratio (float): The proportion of the dataset to allocate to the test set.
    """
    try:
        with open(en_file_path, 'r', encoding='utf-8') as f_en, \
             open(cn_file_path, 'r', encoding='utf-8') as f_cn:
            
            en_lines = f_en.readlines()
            cn_lines = f_cn.readlines()

            if len(en_lines) != len(cn_lines):
                print("Error: The number of lines in the two files does not match.")
                return

            # Pair corresponding lines
            paired_lines = list(zip(en_lines, cn_lines))

            # Shuffle the pairs
            random.shuffle(paired_lines)

            # Calculate the split index
            total_lines = len(paired_lines)
            test_size = int(total_lines * test_ratio)
            train_size = total_lines - test_size

            # Split the data
            train_pairs = paired_lines[:train_size]
            test_pairs = paired_lines[train_size:]

            # Unzip the pairs
            train_en, train_cn = zip(*train_pairs)
            test_en, test_cn = zip(*test_pairs)

            # --- Print Training Data ---
            print("#" * 20)
            print("# Training Data (EN)")
            print("#" * 20)
            with open('./c2e/train.en', 'w', encoding='utf-8') as f:
                f.write(''.join(train_en))

            print("\n" + "#" * 20)
            print("# Training Data (CN)")
            print("#" * 20)
            with open('./c2e/train.cn', 'w', encoding='utf-8') as f:
                f.write(''.join(train_cn))

            # --- Print Testing Data ---
            print("\n" + "#" * 20)
            print("# Testing Data (EN)")
            print("#" * 20)
            with open('./c2e/test.en', 'w', encoding='utf-8') as f:
                f.write(''.join(test_en))

            print("\n" + "#" * 20)
            print("# Testing Data (CN)")
            print("#" * 20)
            with open('./c2e/test.cn', 'w', encoding='utf-8') as f:
                f.write(''.join(test_cn))
                
            print(f"\nSuccessfully split the data.")
            print(f"Total sentences: {total_lines}")
            print(f"Training set size: {len(train_en)} sentences")
            print(f"Testing set size: {len(test_en)} sentences")


    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Define the paths to your files
    # Make sure these paths are correct for your environment
    en_file = 'c2e/t.en'
    cn_file = 'c2e/t.cn'
    
    # Run the split function
    split_data(en_file, cn_file)
