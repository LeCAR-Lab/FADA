def get_index_of_a_in_b(a_names, b_names):
    indexes = []
    for name in a_names:
        assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
        indexes.append(b_names.index(name))
    return indexes
