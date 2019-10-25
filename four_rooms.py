#! /usr/bin/env python
import argparse

import imageio
import numpy as np
import skimage.draw
import skimage.transform
from gym.utils.seeding import np_random
from torch.utils.data import Dataset
import torch
import itertools


def adjacent_sign(k):
    start = max(k - 1, -1)
    stop = min(k + 2, 2)
    return tuple(range(start, stop))


def adjacent_to(k):
    for v in itertools.product(*map(adjacent_sign, k)):
        if np.max(np.abs(np.array(v) - k)) == 1 and v != (0, 0):
            yield v


def get_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class FourRooms(Dataset):
    def __init__(self, room_size: int, distance: float, downscale: int = 5):
        self.downscale_factor = downscale
        self.random = None
        self.size = room_size
        assert room_size % 10 == 0
        self.distance = distance
        self.empty_rooms = np.zeros((room_size + 1, room_size + 1))
        scale = room_size // 10
        mid = room_size // 2
        for start in [0, 3, 5, 8]:
            a = start * scale
            b = (start + 2) * scale
            print("a", a)
            print("b", b)
            rr, cc, val = skimage.draw.line_aa(a, mid, b, mid)
            self.empty_rooms[rr, cc] = val
            rr, cc, val = skimage.draw.line_aa(mid, a, mid, b)
            self.empty_rooms[rr, cc] = val

    def __getitem__(self, index):
        self.random, _ = np_random(index)

        start = 4 * self.random.random(2) - 2  # scale to [-2, 2]
        points = list(
            self.generate_points(
                pos=start,
                distance=self.distance * 4 / self.size,
                explored=set(),
                start=start,
            )
        )
        scaled_points = list(map(self.scale, points))
        x1 = self.draw_points(scaled_points[0], array=np.zeros_like(self.empty_rooms))
        x2 = self.draw_points(scaled_points[-1], array=np.zeros_like(self.empty_rooms))
        x = np.stack([x1, x2, self.empty_rooms], axis=0)
        y = self.draw_lines(*scaled_points, array=np.zeros_like(self.empty_rooms))
        return torch.tensor(x), torch.tensor(y)

    def downscale(self, a):
        return self.downscale_factor * skimage.transform.downscale_local_mean(
            a, (self.downscale_factor, self.downscale_factor)
        )

    @staticmethod
    def draw_points(*points, array):
        for point in points:
            rr, cc = skimage.draw.circle(*point, 1)
            # i, j = array.shape
            # condition = np.logical_and(rr < i, cc < j)
            # rr = rr[condition]
            # cc = cc[condition]
            array[rr, cc] = 1
        return array

    @staticmethod
    def draw_lines(*points, array):
        for p1, p2 in zip(points, points[1:]):
            rr, cc, val = skimage.draw.line_aa(*p1, *p2)
            array[rr, cc] = val
        return array

    def scale(self, p):
        """ scale to array size """
        # virtual graph is 4 x 4 [(-2, -2), (-2, 2)]
        return np.round((np.array(p) + 2) * self.size / 4).astype(int)

    def generate_points(self, pos, distance, explored, start):
        yield pos

        def is_door(*n):
            return 0 in n

        node = np.sign(pos).astype(int)
        adjacent_nodes = list(adjacent_to(node))
        start_node = np.sign(start).astype(int)
        if tuple(-start_node) in adjacent_nodes:
            # always terminate at node opposite start
            new_node = -start_node
        else:

            def filtered_nodes():
                for n in adjacent_nodes:
                    if n not in explored:
                        if is_door(*n):
                            if get_distance(*n, *pos) <= distance:
                                yield n
                        else:
                            far_corner = 2 * np.array(n)
                            if get_distance(*far_corner, *pos) > distance:
                                yield n

            adjacent_nodes = list(filtered_nodes())
            if adjacent_nodes:
                new_node = np.array(
                    adjacent_nodes[self.random.choice(len(adjacent_nodes))]
                )
            else:
                new_node = node
            if is_door(*new_node):
                # recurse at doors
                yield from self.generate_points(
                    pos=new_node,
                    distance=distance - get_distance(*new_node, *pos),
                    start=start,
                    explored=explored | set(adjacent_nodes) | {tuple(node)},
                )
                return

        # terminate at rooms
        for i in range(10):
            if is_door(*node):
                # randomly sample in 180 oriented toward new room
                heading = np.arctan2(*(new_node - node))
                rad = heading + (self.random.random() - 0.5) * np.pi
            else:
                # randomly sample in 360
                rad = self.random.random() * 2 * np.pi
            final_pos = pos + np.array([-np.sin(rad), np.cos(rad)]) * (distance - 1e-5)
            in_world = np.max(np.abs(final_pos)) < 2
            in_room = np.all(np.sign(final_pos) == new_node)
            if not in_world or not in_room:
                edges = np.vstack(
                    [
                        2 * node,  # outer wall
                        np.zeros(2),  # inner wall
                        node - distance,  # in range
                        node + distance,  # in range
                    ]
                )
                edges.sort(axis=0)
                _, low, high, _ = edges
                final_pos = self.random.uniform(low, high)
            if get_distance(*final_pos, *pos) <= distance:
                if np.all(new_node == -start_node):  # opposite room
                    # in the opposite room, we need to make sure that it wouldn't
                    # have been quicker to take the other way around the loop

                    def compute_advantage(x, n):
                        door1, door2 = adjacent_to(n)
                        distance1 = get_distance(*door1, *x)
                        distance2 = get_distance(*door2, *x)
                        advantage = distance1 - distance2
                        if door2 == tuple(-node):
                            return -advantage
                        return advantage

                    current_advantage = compute_advantage(final_pos, new_node)
                    initial_advantage = compute_advantage(start, start_node)
                    if current_advantage + initial_advantage < 0:
                        continue

                yield final_pos
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--distance", type=int, default=53.03300858899107)
    parser.add_argument("--seed", type=int, default=0)
    np.set_printoptions(linewidth=1000, threshold=1000)
    dataset = FourRooms(**vars(parser.parse_args()))
