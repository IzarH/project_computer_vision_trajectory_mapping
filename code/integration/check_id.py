def find_consecutive_id_index(id_list, target_id, consecutive_count=40):
    """
    Finds the last index of the first occurrence of `target_id` with `consecutive_count` consecutive appearances.
    #skips the first 40!!!
    Args:
        id_list (list): List of IDs to search through.
        target_id (int): The ID to find consecutive occurrences of.
        consecutive_count (int): Number of consecutive occurrences to look for.

    Returns:
        int: The ending index of the first occurrence of `target_id` with `consecutive_count` consecutive appearances.
              Returns -1 if not found.
    """
    count = 0
    start_index = -1

    for index, id_value in enumerate(id_list[consecutive_count:]):
        if id_value == target_id:
            if count == 0:
                start_index = index  # Record the starting index of the sequence
            count += 1
            
            if count == consecutive_count:
                return index+consecutive_count  # Return the last index of the consecutive sequence
        else:
            count = 0
            start_index = -1  # Reset start index if the sequence breaks

    return -1

# Example usage
#ids = [1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2,2,2,2,2,2,2,2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 7, 8]  # Example list
#target_id = 2
#index = find_consecutive_id_index(ids, target_id)

#if index != -1:
#    print(f"The last index where ID {target_id} appears consecutively 20 times is: {index}")
#else:
#    print(f"ID {target_id} does not appear consecutively 20 times.")