from itertools import chain, combinations


def powerset(iterable):
    """
    Returns the powerset of the given iterable
    :param iterable: The iterable to find the powerset of
    :return: The powerset of the given iterable
    """

    s = list(iterable)
    # Return the powerset of the given iterable in list form. chain.from_iterable() is used to flatten the list
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_subset_counts(data, min_support):
    """
    Returns the counts of each subset in the given data
    :param data: The data to get the subset counts for
    :param min_support: The minimum support for a subset to be considered frequent
    :return: The counts of each subset in the given data 
    """
    subset_counts = {}

    # Count the number of times each subset appears in the data
    for transaction in data:
        # Get the powerset of the transaction
        for subset in powerset(transaction):
            if subset:
                # Sort the subset so that the same subset with different orderings are counted together
                subset_tuple = tuple(sorted(subset))
                subset_counts[subset_tuple] = subset_counts.get(subset_tuple, 0) + 1

    # Filter out subsets that don't meet the minimum support
    subset_counts = {subset: count for subset, count in subset_counts.items() if count >= min_support}

    return subset_counts


def generate_association_rules(subset_counts, min_confidence):
    """
    Generates the association rules from the given subset counts
    :param subset_counts: The counts of each subset in the data
    :param min_confidence: The minimum confidence for a rule to be considered strong
    :return: A list of association rules and their confidence
    """
    association_rules = []

    for subset, count in subset_counts.items():
        if len(subset) > 1:
            for item in subset:
                antecedent = tuple(sorted(set(subset) - {item}))
                if antecedent in subset_counts:
                    rule = (antecedent, (item,))
                    confidence = count / subset_counts[antecedent]
                    if confidence >= min_confidence:
                        association_rules.append((rule, confidence))

    return association_rules

def apriori(data, min_support, min_confidence):
    """
    Runs the Apriori algorithm on the given data.
    :param data: The data to run the algorithm on
    :param min_support: The minimum support for a subset to be considered frequent
    :param min_confidence: The minimum confidence for a rule to be considered strong
    :return: A list of association rules and their confidence
    """
    # Get the counts of each subset
    subset_counts = get_subset_counts(data, min_support)

    # Generate the association rules
    association_rules = generate_association_rules(subset_counts, min_confidence)

    return subset_counts, association_rules