def scoreResponse(correct: str, response: str) -> float:
    if not correct:
        return 0.0
    
    hammingDistance = sum(c != r for c, r in zip(correct, response))
    hammingDistance += abs(len(correct) - len(response)) # responses too long or too short have all excess/nonexistent characters counted as incorrect
    
    return 1 - (hammingDistance/max(len(correct), len(response))) # in [0, 1]