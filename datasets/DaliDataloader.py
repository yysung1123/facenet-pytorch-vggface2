"""The original code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/data_loader.py

    The code was modified to speed up the triplet generation process by bypassing the dataframe.loc operation,
     generate batches according to a set amount of human identities (classes) per triplet batch, and to
     support .png, .jpg, and .jpeg files.
"""


import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy


class FedEBPTripletFaceDataset(Dataset):
    def __init__(self, root_dir, ebp_root_dir, csv_name, num_triplets, epoch, classes, num_human_identities_per_batch=32,
                 triplet_batch_size=320, training_triplets_path=None, transform=None):
        """
        Args:

        root_dir: Absolute path to dataset.
        csv_name: Path to csv file containing the image paths inside the training dataset folder.
        num_triplets: Number of triplets required to be generated.
        epoch: Current epoch number (used for saving the generated triplet list for this epoch).
        num_generate_triplets_processes: Number of separate Python processes to be created for the triplet generation
                                          process. A value of 0 would generate a number of processes equal to the
                                          number of available CPU cores.
        num_human_identities_per_batch: Number of set human identities per batch size.
        triplet_batch_size: Required number of triplets in a batch.
        training_triplets_path: Path to a pre-generated triplet numpy file to skip the triplet generation process (Only
                                 will be used for one epoch).
        transform: Required image transformation (augmentation) settings.
        """

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #  VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #  forcing the 'name' column as being of type 'int' instead of type 'object')
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        self.df = self.df[self.df['class'].isin(classes)]
        self.df = self.df.reset_index()
        self.root_dir = root_dir
        self.ebp_root_dir = ebp_root_dir
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.epoch = epoch
        self.transform = transform

        # Modified here to bypass having to use pandas.dataframe.loc for retrieving the class name
        #  and using dataframe.iloc for creating the face_classes dictionary
        df_dict = self.df.to_dict()
        self.df_dict_class_name = df_dict["name"]
        self.df_dict_id = df_dict["id"]
        self.df_dict_class_reversed = {value: key for (key, value) in df_dict["class"].items()}

        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets()
        '''
        else:
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path)
        '''

    def make_dictionary_for_face_class(self):
        """
            face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        face_classes = dict()
        for idx, label in enumerate(self.df['class']):
            if label not in face_classes:
                face_classes[label] = []
            # Instead of utilizing the computationally intensive pandas.dataframe.iloc() operation
            face_classes[label].append(self.df_dict_id[idx])

        return face_classes

    def generate_triplets(self):
        triplets = []
        classes = self.df['class'].unique()
        face_classes = self.make_dictionary_for_face_class()

        print("\nGenerating {} triplets ...".format(self.num_triplets))
        num_training_iterations_per_process = self.num_triplets / self.triplet_batch_size
        progress_bar = tqdm(range(int(num_training_iterations_per_process)))  # tqdm progress bar does not iterate through float numbers

        for training_iteration in progress_bar:

            """
            For each batch:
                - Randomly choose set amount of human identities (classes) for each batch

                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """
            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while len(face_classes[pos_class]) < 2:
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)

                # Instead of utilizing the computationally intensive pandas.dataframe.loc() operation
                pos_name_index = self.df_dict_class_reversed[pos_class]
                pos_name = self.df_dict_class_name[pos_name_index]

                neg_name_index = self.df_dict_class_reversed[neg_class]
                neg_name = self.df_dict_class_name[neg_name_index]

                if len(face_classes[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:
                    ianc = np.random.randint(0, len(face_classes[pos_class]))
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

                    while ianc == ipos:
                        ipos = np.random.randint(0, len(face_classes[pos_class]))

                ineg = np.random.randint(0, len(face_classes[neg_class]))

                triplets.append(
                    [
                        face_classes[pos_class][ianc],
                        face_classes[pos_class][ipos],
                        face_classes[neg_class][ineg],
                        pos_class,
                        neg_class,
                        pos_name,
                        neg_name
                    ]
                )

        '''
        print("Saving training triplets list in 'datasets/generated_triplets' directory ...")
        np.save('datasets/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                self.epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ),
            triplets
        )
        print("Training triplets' list Saved!\n")
        '''

        return triplets

    # Added this method to allow .jpg, .png, and .jpeg image support
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        elif os.path.exists(path + '.jpeg'):
            return path + '.jpeg'
        else:
            raise RuntimeError('No file "{}" with extension .png or .jpg or .jpeg'.format(path))

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))
        ebp_anc_img = self.add_extension(os.path.join(self.ebp_root_dir, str(pos_name), str(anc_id)))
        ebp_pos_img = self.add_extension(os.path.join(self.ebp_root_dir, str(pos_name), str(pos_id)))
        ebp_neg_img = self.add_extension(os.path.join(self.ebp_root_dir, str(neg_name), str(neg_id)))

        # Modified to open as PIL image in the first place
        anc_img = np.fromfile(anc_img, dtype = np.uint8)
        pos_img = np.fromfile(pos_img, dtype = np.uint8)
        neg_img = np.fromfile(neg_img, dtype = np.uint8)
        ebp_anc_img = np.fromfile(ebp_anc_img, dtype = np.uint8)
        ebp_pos_img = np.fromfile(ebp_pos_img, dtype = np.uint8)
        ebp_neg_img = np.fromfile(ebp_neg_img, dtype = np.uint8)


        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'ebp_anc_img': ebp_anc_img,
            'ebp_pos_img': ebp_pos_img,
            'ebp_neg_img': ebp_neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)

class InputIterator(object):
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        anc_batch = []
        pos_batch = []
        neg_batch = []
        ebp_anc_batch = []
        ebp_pos_batch = []
        ebp_neg_batch = []

        if self.i >= len(self.dataset) // self.batch_size * self.batch_size:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            sample = self.dataset[self.i]
            anc_batch.append(sample['anc_img'])
            pos_batch.append(sample['pos_img'])
            neg_batch.append(sample['neg_img'])
            ebp_anc_batch.append(sample['ebp_anc_img'])
            ebp_pos_batch.append(sample['ebp_pos_img'])
            ebp_neg_batch.append(sample['ebp_neg_img'])
            self.i += 1

        return (anc_batch, pos_batch, neg_batch, ebp_anc_batch, ebp_pos_batch, ebp_neg_batch)

    def __len__(self):
        return len(self.dataset) // self.batch_size

def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data, image_size):
    def _pipe(images, ebp_images, image_size):
        images = fn.resize(images, resize_x=image_size, resize_y=image_size)
        ebp_images = fn.resize(ebp_images, resize_x=image_size, resize_y=image_size)
        images = fn.cat(images, ebp_images, axis=2)

        images = fn.flip(images)
        images = fn.rotate(images, angle=5.0 * np.pi / 180, size=(224,224))
        output = fn.crop_mirror_normalize(images,
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          mean=[0.6068 * 255, 0.4517 * 255, 0.3800 * 255, 0, 0, 0],
                                          std=[0.2492 * 255, 0.2173 * 255, 0.2082 * 255, 255, 255, 255])
        return output

    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        anc_imgs, pos_imgs, neg_imgs, ebp_anc_imgs, ebp_pos_imgs, ebp_neg_imgs = fn.external_source(source=external_data, num_outputs=6)
        anc_imgs = fn.decoders.image(anc_imgs, device="mixed")
        pos_imgs = fn.decoders.image(pos_imgs, device="mixed")
        neg_imgs = fn.decoders.image(neg_imgs, device="mixed")
        ebp_anc_imgs = fn.decoders.image(ebp_anc_imgs, device="mixed")
        ebp_pos_imgs = fn.decoders.image(ebp_pos_imgs, device="mixed")
        ebp_neg_imgs = fn.decoders.image(ebp_neg_imgs, device="mixed")
        anc_imgs = _pipe(anc_imgs, ebp_anc_imgs, image_size)
        pos_imgs = _pipe(pos_imgs, ebp_pos_imgs, image_size)
        neg_imgs = _pipe(neg_imgs, ebp_neg_imgs, image_size)
        pipe.set_outputs(anc_imgs, pos_imgs, neg_imgs)
    return pipe

def get_dataloader(root_dir, ebp_root_dir, csv_name, image_size, triplet_batch_size, iterations_per_epoch, num_human_identities_per_batch, epoch, classes_split, num_threads=16):
    dataset = FedEBPTripletFaceDataset(
        root_dir=root_dir,
        ebp_root_dir=ebp_root_dir,
        csv_name=csv_name,
        num_triplets=iterations_per_epoch * triplet_batch_size,
        num_human_identities_per_batch=num_human_identities_per_batch,
        triplet_batch_size=triplet_batch_size,
        epoch=epoch,
        classes=classes_split,
        training_triplets_path=None,
        transform=None
    )
    ii = InputIterator(batch_size=triplet_batch_size, dataset=dataset)
    pipe = ExternalSourcePipeline(batch_size=triplet_batch_size, num_threads=num_threads, device_id=0, external_data=ii, image_size=image_size)
    pii = PyTorchIterator(pipe, ["anc_img", "pos_img", "neg_img"], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    return pii
