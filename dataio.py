from torchvision.datasets import ImageFolder

if __name__ == "__main__":
    face_folder = ImageFolder('./AnimeDataset/faces')

    img = face_folder.loader()
    print(img)