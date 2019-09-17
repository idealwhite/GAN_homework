from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from torch.utils.data import Dataset
import numpy as np
import torch

transform = transforms.Compose([transforms.RandomRotation(20, expand=True),
                              transforms.CenterCrop([64,64]),
                              transforms.RandomHorizontalFlip(0.5),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5]*3, [0.5]*3)])

face_folder = ImageFolder('./AnimeDataset', transform=transform)


def get_negative_image_idx_tags(truth_idx):
    positive_tags = id2tags(truth_idx)

    # get all negative idx
    negative_idx_pool = []
    negative_tags_pool = []
    for tags, pool in tags2ids_pool.items():
        if tags != positive_tags:
            negative_idx_pool += pool
            negative_tags_pool.append(tags)

    # get a random negative image idx from all negative labels
    random_selection_image = np.random.randint(0, len(negative_idx_pool))
    random_selection_tags = np.random.randint(0, len(negative_tags_pool))


    return negative_idx_pool[random_selection_image], negative_tags_pool[random_selection_tags]


def get_tag_tools():
    # id 2 tag
    with open('./AnimeDataset/extra_data/tags.csv', 'r') as f:
        id2tags = {int(line.split(',')[0]): line.split(',')[1].strip() for line in f.readlines()}

    # tag 2 vector
    hair_tags = set()
    eye_tags = set()
    for idx in id2tags:
        color1, _, color2, _ = id2tags[idx].split(' ')
        hair_tags.add(color1)
        eye_tags.add(color2)
    hairtag2id = {tag: idx for idx, tag in enumerate(list(hair_tags))}
    eyetag2id = {tag: idx for idx, tag in enumerate(list(eye_tags))}

    def id2tags_envelope(idx):
        return id2tags[idx]

    def tags2vec(tags):
        color1, _, color2, _ = tags.split(' ')
        vector = torch.zeros(len(hairtag2id)+len(eyetag2id))
        vector[hairtag2id[color1]] = 1
        vector[len(hairtag2id) + eyetag2id[color2]] = 1
        return vector
    return id2tags_envelope, tags2vec, id2tags

id2tags, tags2vec, id2tags_dict = get_tag_tools()

tags2ids_pool = {}
for id, tags in id2tags_dict.items():
    tags2ids_pool[tags] = [id] if tags not in tags2ids_pool else tags2ids_pool[tags] + [id]

def tags2vec_batch(tags, device):
    vec = torch.stack([tags2vec(tag) for tag in tags], 0).to(device)
    return vec


class FolderDataset(Dataset):
    def __init__(self, device, image_folder, conditions=False):
        self.image_folder = image_folder
        self.conditions = conditions
        self.device = device

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, item):
        if self.conditions:
            image_tensor, idx = self.image_folder[item]
            csv_idx = int(self.image_folder.classes[idx])

            negative_csv_idx, negative_tags = get_negative_image_idx_tags(csv_idx)
            negative_image_tensor, _ = self.image_folder[self.image_folder.class_to_idx[str(negative_csv_idx)]]
            idx_vec = tags2vec(id2tags(csv_idx))
            negative_vec = tags2vec(negative_tags)
            return tuple([image_tensor.to(self.device), idx_vec.to(self.device), \
                          negative_image_tensor.to(self.device), negative_vec.to(self.device)])
        else:
            return tuple([self.image_folder[item][0].to(self.device)])


def get_condition_image_dataset(device, file_path='./AnimeDataset/extra_data/', transform=transform):
    '''
    Return a tensordataset, which generate [image], [embedding] record at each time.
    '''
    image_folder = ImageFolder(file_path+'images/', transform=transform)

    return FolderDataset(device, image_folder, conditions=True)


def get_noise_batch(batch_size, dim_noise, device):
    noise_batch = torch.normal(0, 1, [batch_size, dim_noise]).to(device)
    return noise_batch

def get_fake_batch(generator, batch_size, dim_noise, device):
    noise_batch = get_noise_batch(batch_size, dim_noise, device)
    batch_fake = generator(noise_batch).to(device)
    return batch_fake

def get_fake_batch_conditional(generator, batch_condition, batch_size, dim_noise, device):
    noise_batch = get_noise_batch(batch_size, dim_noise, device)
    batch_fake = generator(noise_batch, batch_condition).to(device)

    return batch_fake

if __name__ == "__main__":
    file_path = './AnimeDataset/extra_data/'
    image_folder = ImageFolder(file_path + 'images/', transform=transform)
