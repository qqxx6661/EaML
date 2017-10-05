#!/usr/bin/env python3
# coding=utf-8

def relative_position(cood):
    """
    :type cood: List[x_start, y_start, x_end, y_end]
    :rtype: int
    """
    # 输入两点坐标的list，其实就是矩形的左上角和右下角坐标。求该点在画面内的相对位置（左右来看）。视频分辨率为640x480。
    # 举例：该点x坐标是320，那他的相对位置就是0
    return ((0.5 * (cood[0] + cood[2])) - 320) / 320


if __name__ == "__main__":
    print(relative_position([480, 58, 480, 284]))