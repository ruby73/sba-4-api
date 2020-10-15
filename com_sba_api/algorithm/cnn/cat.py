import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import array, zeros_like

class Cat:
    def show(self):
        image_path = tf.keras.utils.get_file('cat.jpg', 'http://bit.ly/33U6mH9' )
        image = plt.imread(image_path)
        titles = ['RGB Image', 'Red channel', 'Green channel', 'Blue channel']
        colors = range(-1, 3)
        fig, axes = plt.subplots(1, 4, figsize = (13, 3))
        objs = zip(axes, titles, colors)

        for ax, title, color in objs:
            ax.imshow(self.channel(image, color))
            ax.set_title(title)
            ax.set_xticks(())
            ax.set_yticks(())
        plt.show()

    @staticmethod
    def channel(image, color):
        if color not in (0, 1, 2) : return image
        c = image[..., color]
        z = zeros_like(c)
        return array([(c,z,z,), (z,c,z), (z,z,c)][color]).transpose(1, 2, 0)

cat = Cat()
cat.show()