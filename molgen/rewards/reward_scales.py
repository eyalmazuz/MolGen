def mult(value: float, factor: float = 10.0) -> float:
    return factor * value


def minmax(value: float, min_value: float, max_value: float, new_min: float = 0.0, new_max: float = 1.0) -> float:
    if max_value == min_value:
        return new_min
    if value < min_value:
        return new_min
    if value > max_value:
        return new_max

    return new_min + ((value - min_value) / (max_value - min_value)) * (new_max - new_min)
