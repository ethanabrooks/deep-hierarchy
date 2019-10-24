import itertools

import numpy as np

#########
#   #   #
# 1 a 2 #
#   #   #
##b###c##
#   #   #
# 3 d 4 #
#   #   #
#########

# room 1: -1, 1
# room 2: 1, 1
# room 3: -1, -1
# room 4: 1, -1

# door a: 0, 1
# door b: -1, 0
# door c: 1, 0
# door d: 0, -1

size = 100


def adjacent_sign(k):
    start = max(k - 1, -1)
    stop = min(k + 2, 2)
    return tuple(range(start, stop))


def adjacent_to(k1, k2):
    for v in itertools.product(adjacent_sign(k1), adjacent_sign(k2)):
        if 0 in v:  # v is a door
            if v not in [(0, 0), (k1, k2)]:  # no center or self-loops
                yield v


def get_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def in_range(pos, node, distance):
    if 0 in node:  # door
        if not get_distance(*node, *pos) > distance:
            return False
    return np.all(np.sign(pos) == np.sign(node))


def generate_points(pos, distance, explored):
    yield pos
    current_node = np.sign(pos)
    adjacent_doors = adjacent_to(*current_node)
    doors_in_range = [
        d
        for d in adjacent_doors
        if get_distance(*d, *pos) < distance and d not in explored
    ]
    if doors_in_range:
        next_pos = np.random.choice(adjacent_doors)
        yield from generate_points(
            pos=next_pos,
            distance=distance - get_distance(*next_pos, *pos),
            explored=explored | adjacent_doors,
        )
    while True:
        rad = 2 * np.pi * np.random.random()
        next_pos = [np.cos(rad), np.sign(rad)]  # sample next_pos at edge of range
        if np.all(np.abs(next_pos) < 1) and np.sign(next_pos) == current_node:
            break  # next_pos is in same room

        # in case the intersection between the range arc and the room is small,
        # we sample again within the room and accept any point within range
        next_pos = current_node * np.random.random(2)  # sample next_pos in same room
        if get_distance(*pos, *next_pos) < distance:
            break  # next_pos is in range

    yield next_pos


# adjacent = {
#     k: adjacent_to(*k)
#     for k in itertools.product(range(-1, 2), repeat=2)
#     if k != (0, 0)  # no center
# }

# adjacent_hand_coded = {
#     (-1, 1): [(0, 1), (-1, 0)],  # room 1
#     (1, 1): [(0, 1), (1, 0)],  # room 2
#     (-1, -1): [(-1, 0), (0, -1)],  # room 3
#     (1, -1): [(1, 0), (0, -1)],  # room 4
#     (0, 1): [(-1, 0), (1, 0)],  # door a
#     (-1, 0): [(0, 1), (0, -1)],  # door b
#     (1, 0): [(0, 1), (0, -1)],  # door c
#     (0, -1): [(-1, 0), (1, 0)],  # door d
# }
# if set(adjacent.keys()) != set(adjacent_hand_coded.keys()):
#     import ipdb
#
#     ipdb.set_trace()
# for key in adjacent.keys():
#     if set(adjacent_hand_coded[key]) != set(adjacent[key]):
#         import ipdb
#
#         ipdb.set_trace()
