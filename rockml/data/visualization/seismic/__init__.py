""" Copyright 2019 IBM Research. All Rights Reserved.

    - Visualization functions for seismic seisfast.
"""

import os
from random import shuffle
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rockml.data.adapter.seismic.segy.poststack import PostStackDatum


def export_classification_examples(tiles: List[PostStackDatum], path: str, tiles_per_line: int = 10, lines: int = 2):
    """ Save examples of tiles (randomly selected in the list) for each class in a png image.

    Args:
        tiles: list of processed SeismicDatums.
        path: (str) path where to save the example image.
        tiles_per_line: (int) defines how many tiles will be printed per line.
        lines: (int) defines how many lines will be printed.
    """

    labels = [t.label for t in tiles]
    unique = np.unique(labels)
    num_classes = len(unique)

    if tiles[0].features.shape[2] == 1:
        mode = 'L'
        color = 255
    else:
        mode = 'RGB'
        color = (255, 255, 255)

    tile_height = tiles[0].features.shape[0]
    tile_width = tiles[0].features.shape[1]

    # horizontal distance in pixels between tiles
    horizontal_dist = 5
    # vertical distance in pixels between tiles
    vertical_dist = 5
    # distance in pixels between tiles of different classes
    class_dist = 30

    height = (num_classes * tile_height * lines) + (num_classes + 1) * class_dist + \
             (num_classes * lines) * vertical_dist
    width = (tiles_per_line * tile_width) + (tiles_per_line + 1) * horizontal_dist

    new_im = Image.new(mode, (width, height), color=color)

    x = horizontal_dist
    y = class_dist

    step_x = tile_width + horizontal_dist
    step_y = tile_height + vertical_dist
    offset = int(class_dist / 4)

    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', "SourceSansPro-Regular.ttf"), 14)
    draw = ImageDraw.Draw(new_im)

    for idx, c in enumerate(unique):
        selected = [tiles[i] for i in np.where(labels == c)[0]]
        shuffle(selected)
        title = 'Class: ' + str(unique[idx])
        draw.text(xy=(x, y - class_dist + offset), text=title, fill=0, font=font)
        draw = ImageDraw.Draw(new_im)
        for line in range(lines):
            for img in range(tiles_per_line):
                if (tiles_per_line * line) + img >= len(selected):
                    break
                if mode == 'L':
                    temp_img = Image.fromarray(
                        selected[(tiles_per_line * line) + img].features[:, :, 0])
                else:
                    temp_img = Image.fromarray(
                        selected[(tiles_per_line * line) + img].features)
                new_im.paste(temp_img, (x, y))
                x = x + step_x

            y = y + step_y
            x = horizontal_dist

        y = y + class_dist

    new_im.save(path)


def export_segmentation_examples(tiles: List[PostStackDatum], path: str, tiles_per_line: int = 10, lines: int = 2):
    """ Save examples of tiles (randomly selected in the list) and their labels in a png image.

    Args:
        tiles: list of processed SeismicDatums.
        path: (str) path to save the example image.
        tiles_per_line: (int) defines how many tiles will be printed per line.
        lines: (int) defines how many lines will be printed.
    """

    num_classes = len(np.unique([t.label for t in tiles]))

    # Each block consists of 1 line of tiles and another line of their respective labels
    block_lines = 2

    if tiles[0].features.shape[2] == 1:
        mode = 'L'
        color = 255
    else:
        mode = 'RGB'
        color = (255, 255, 255)

    tile_height = tiles[0].features.shape[0]
    tile_width = tiles[0].features.shape[1]

    # horizontal distance in pixels between tiles
    horizontal_dist = 5
    # vertical distance in pixels between tiles and respective labels
    label_dist = 5
    # vertical distance in pixels between different blocks of tiles and labels
    block_dist = 20

    height = (lines * tile_height * block_lines) + (lines + 1) * block_dist + \
             (lines * block_lines) * label_dist
    width = (tiles_per_line * tile_width) + (tiles_per_line + 1) * horizontal_dist
    new_im = Image.new(mode, (width, height), color=color)

    x = horizontal_dist
    y = block_dist

    step_x = tile_width + horizontal_dist
    step_y = tile_height + label_dist

    selected = np.random.choice(tiles, tiles_per_line * lines, replace=False)
    images = np.asarray([datum.features[:, :, :] for datum in selected])
    labels = np.asarray([datum.label[:, :] for datum in selected])
    labels = (labels * 255.0 / num_classes).astype(np.uint8)

    for idx in range(lines):
        ImageDraw.Draw(new_im)
        for img in range(tiles_per_line):
            if mode == 'L':
                temp_img = Image.fromarray(images[(tiles_per_line * idx) + img, :, :, 0])
            else:
                temp_img = Image.fromarray(images[(tiles_per_line * idx) + img, :, :, :])
            temp_mask = Image.fromarray(labels[(tiles_per_line * idx) + img, :, :])
            new_im.paste(temp_img, (x, y))
            new_im.paste(temp_mask, (x, y + step_y))
            x = x + step_x

        x = horizontal_dist
        y = y + (block_lines * tile_height) + label_dist + block_dist

    new_im.save(path)
