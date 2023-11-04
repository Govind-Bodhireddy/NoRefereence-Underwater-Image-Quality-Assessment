from PIL import Image
import numpy as np

def uniform_lbp(img, radius, n_points):
    def get_interpolated_pixel(image, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0
        pixel = (1 - dx) * (1 - dy) * image[y0, x0] + dx * (1 - dy) * image[y0, x1] + \
                (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1]
        return pixel

    img_gray = img.convert("L")
    img_array = np.array(img_gray)

    height, width = img_array.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            lbp_code = 0
            center_pixel = img_array[y, x]

            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x_i = x + radius * np.cos(angle)
                y_i = y - radius * np.sin(angle)

                pixel = get_interpolated_pixel(img_array, x_i, y_i)
                if pixel >= center_pixel:
                    lbp_code |= (1 << i)

            # Convert LBP code to a uniform pattern
            uniform_lbp_code = 0
            for i in range(n_points):
                bit1 = (lbp_code >> i) & 1
                bit2 = (lbp_code >> ((i + 1) % n_points)) & 1
                uniform_lbp_code |= (bit1 ^ bit2) << i

            lbp_image[y, x] = uniform_lbp_code

    return Image.fromarray(lbp_image)

