import cv2
import torch


def Vertical_Projection(src: str) -> list:
    seg_img: list = []
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)  # 二值化
    (thresh_h, thresh_w) = thresh.shape

    # 纵向垂直投影
    line = torch.zeros(thresh_h)
    for j in range(0, thresh_h):
        for i in range(0, thresh_w):
            if thresh[j, i] != 255:
                line[j] += 1
    flip_flop: bool = False
    white: int = 0
    black: int = 0
    line_point: list = []
    for i in range(0, thresh_h):
        if line[i] > 0:
            if not flip_flop:
                line_point.append(white)
                white = 0
                flip_flop = True
            black += 1
        else:
            if flip_flop:
                line_point.append(black)
                black = 0
                flip_flop = False
            white += 1
    num_line = len(line_point) / 2
    line_point.append(thresh_h - sum(line_point))
    img_line = torch.from_numpy(thresh)
    img_line = torch.split(img_line, split_size_or_sections=line_point, dim=0)

    # 列垂直投影
    for i in range(0, int(num_line)):
        image = img_line[2 * i + 1].numpy()
        (image_h, image_w) = image.shape
        column = torch.zeros(image_w)
        for i in range(0, image_w):
            for j in range(0, image_h):
                if image[j, i] != 255:
                    column[i] += 1
        flip_flop: bool = False
        white: int = 0
        black: int = 0
        column_point: list = []
        for i in range(0, image_w):
            if column[i] > 0:
                if not flip_flop:
                    column_point.append(white)
                    white = 0
                    flip_flop = True
                black += 1
            else:
                if flip_flop:
                    column_point.append(black)
                    black = 0
                    flip_flop = False
                white += 1
        num_column = len(column_point) / 2
        column_point.append(image_w - sum(column_point))
        img_column = torch.from_numpy(image)
        img_column = torch.split(img_column, split_size_or_sections=column_point, dim=1)

        # 对图片进行预处理增加长宽,防止改变大小时产生过大的形变
        for i in range(0, int(num_column)):
            (h, w) = img_column[2 * i + 1].shape
            if h > w:
                num = (h - w) // 2
                front = torch.full([h, num], fill_value=255, dtype=torch.float32)
                behind = torch.full([h, num], fill_value=255, dtype=torch.float32)
                im = img_column[2 * i + 1]
                im = torch.cat([front, im], dim=1)
                im = torch.cat([im, behind], dim=1)
                im = im.numpy()
                seg_img.append(im)
            else:
                num = (w - h) // 2
                front = torch.full([num, w], fill_value=255, dtype=torch.float32)
                behind = torch.full([num, w], fill_value=255, dtype=torch.float32)
                im = img_column[2 * i + 1]
                im = torch.cat([front, im], dim=0)
                im = torch.cat([im, behind], dim=0)
                im = im.numpy()
                seg_img.append(im)
    return seg_img
