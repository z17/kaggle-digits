import matplotlib.pyplot as plt
import pandas as pd


def show_image_from_list(i, images):
    show_image(images.iloc[i])
    return


def show_image(image):
    img = image.as_matrix()
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='binary')
    plt.show()
    return


def save_results(results, path):
    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns = ['Label']
    df.to_csv(path, header=True)
    return


def convert_result(v):
    for index, val in enumerate(v):
        if val == 1:
            return index
    return 0
