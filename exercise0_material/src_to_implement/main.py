import pattern
import generator
import numpy as np
from PIL import Image
from src_to_implement.generator import ImageGenerator

# c = pattern.Checker(400, 50)
# c.draw()
# c.show()

# c = pattern.Circle(200, 50, (100,100))
# c.draw()
# c.show()

# c = pattern.Spectrum(500)
# c.draw()
# c.show()


ig = ImageGenerator(
    file_path='./exercise_data',
    label_path='./Labels.json',
    batch_size=10,
    image_size=[30, 30]
)

ig2 = ImageGenerator(
    file_path='./exercise_data',
    label_path='./Labels.json',
    batch_size=15,
    image_size=[30, 30]
)

x, y = ig.next()
x2, y2 = ig2.next()

print('al')