def scale_box(box_coord, width_scale, height_scale):
    x1, y1, x2, y2 = box_coord
    return [x1 * width_scale, y1 * height_scale, x2 * width_scale, y2 * height_scale]


def get_color_dict(config_color_dict, default_color, label_dict):
    for key in list(config_color_dict.keys()):
        if len(config_color_dict[key]) == 0:
            config_color_dict[key] = default_color
        elif len(config_color_dict[key]) != 3:
            raise ValueError(
                f'Expected color for "{label_dict[key]}" to be in BGR color format. '
                f"Got {config_color_dict[key]} instead."
            )

    return config_color_dict
