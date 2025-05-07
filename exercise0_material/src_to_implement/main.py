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


ig = ImageGenerator(
    file_path='./data/exercise_data',
    label_path='./data/Labels.json',
    batch_size=5,
    image_size=[64, 64]
)

x,y = ig.next()

for i in range(x.shape[0]):
    img = x[i]
    Image.fromarray(img).show()