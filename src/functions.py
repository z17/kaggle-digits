import matplotlib.pyplot as plt


def showimage(i, train_images):
    img = train_images.iloc[i].as_matrix()
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='binary')
    plt.show()
    return
