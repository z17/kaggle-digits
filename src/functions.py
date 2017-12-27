import matplotlib.pyplot as plt
import pandas as pd


def showimage(i, images):
    img = images.iloc[i].as_matrix()
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
