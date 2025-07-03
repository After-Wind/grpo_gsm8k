import re

def extract_answer(text):
    # First, try to find text after the first occurrence of ####
    if "####" in text:
        # Find the first occurrence and get text after it
        idx = text.find("####")
        after_hashes = text[idx + 4:]

        # Extract all numbers from the text after ####
        numbers = re.findall(r"\d+", after_hashes)
        if numbers:
            # Return the first number found after ####
            return int(numbers[0])

    # If no #### found, extract the last number in the entire text
    all_numbers = re.findall(r"\d+", text)
    if all_numbers:
        return int(all_numbers[-1])

    return None
