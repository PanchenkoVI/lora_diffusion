from PIL import Image
import numpy as np


class PadResize:
    def __init__(self, size=512, mode="reflect"):
        """
        mode: 'reflect' (отражение), 'edge' (повторение), 'wrap' (обертка)
        """

        self.size = size
        self.mode = mode

    def __call__(self, image):
        w, h = image.size

        if w == h:
            return image.resize((self.size, self.size), Image.LANCZOS)

        max_side = max(w, h)

        if w < max_side:
            pad_left = (max_side - w) // 2
            pad_right = max_side - w - pad_left
            padding = (pad_left, 0, pad_right, 0)
        else:
            pad_top = (max_side - h) // 2
            pad_bottom = max_side - h - pad_top
            padding = (0, pad_top, 0, pad_bottom)

        img_array = np.array(image)

        if self.mode == "reflect":
            # Отражение краев (самый естественный вид)
            padded_array = np.pad(
                img_array,
                ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
                mode="reflect",
            )
        elif self.mode == "edge":
            # Повторение крайних пикселей
            padded_array = np.pad(
                img_array,
                ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
                mode="edge",
            )
        elif self.mode == "wrap":
            # Обертка (циклическое повторение)
            padded_array = np.pad(
                img_array,
                ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
                mode="wrap",
            )
        else:
            # Fallback to edge
            padded_array = np.pad(
                img_array,
                ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
                mode="edge",
            )

        padded_image = Image.fromarray(padded_array.astype("uint8"))
        return padded_image.resize((self.size, self.size), Image.LANCZOS)
