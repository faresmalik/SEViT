import os 


from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# DataLoader and Dataset (Clean Samples)
def data_loader( root_dir, image_size = (224,224), batch_size= 30, train_dir = 'training',test_dir = 'testing',  vald_dir = 'validation'): 
        """
        Class to create Dataset and DataLoader from Image folder. 
        Args: 
            image_size -> size of the image after resize 
            batch_size 
            root_dir -> root directory of the dataset (downloaded dataset) 

        return: 
            dataloader -> dict includes dataloader for train/test and validation 
            dataset -> dict includes dataset for train/test and validation 
        
        """

        dirs = {'train' : os.path.join(root_dir,train_dir),
                'valid' : os.path.join(root_dir,vald_dir), 
                'test' : os.path.join(root_dir,test_dir) 
                }


        data_transform = {
                'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation (20),
                transforms.Resize(image_size),
                transforms.RandomAffine(degrees =0,translate=(0.1,0.1)),
                transforms.ToTensor()
                ]), 

                'valid': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
                ]), 

                'test' : transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
                ])
                }


        image_dataset = {x: ImageFolder(dirs[x], transform= data_transform[x]) 
                                        for x in ('train', 'valid','test')}

        data_loaders= {x: DataLoader(image_dataset[x], batch_size= batch_size,
                                shuffle=True, num_workers=12) for x in ['train']}

        data_loaders['test'] = DataLoader(image_dataset['test'], batch_size= batch_size,
                                shuffle=False, num_workers=12, drop_last=True)
        
        data_loaders['valid'] = DataLoader(image_dataset['valid'], batch_size= batch_size,
                                shuffle=False, num_workers=12, drop_last=True)

        dataset_size = {x: len(image_dataset[x]) for x in ['train', 'valid','test']}


        print ([f'number of {i} images is {dataset_size[i]}' for i in (dataset_size)])

        class_idx= image_dataset['test'].class_to_idx
        print (f'Classes with index are: {class_idx}')

        class_names = image_dataset['test'].classes
        print(class_names)
        return data_loaders, image_dataset

#Dataloader and Dataset (Adversarial Samples)
def data_loader_attacks(root_dir, attack_name ,image_size = (224,224), batch_size = 30): 
       
        """
        Class to create Dataset and DataLoader from Image folder for adversarial samples generated. 
        Args: 
            root _dir: root directory of generated adversarial samples.
            attack_name: attack name that has folder in root_dir.
            image_size : size of the image after resize (224,224)
            batch_size

        return: 
            dataloader : dataloader for the attack
            dataset :  dataset for attack 
        
        """

        dirs = os.path.join(root_dir, f'Test_attacks_{attack_name}')
        data_transform = transforms.Compose([transforms.Resize(image_size),
                                                transforms.ToTensor()]
                                                )
        
        image_dataset = ImageFolder(dirs, transform= data_transform)

        data_loaders =DataLoader(image_dataset, batch_size= batch_size,
                              shuffle=False, num_workers=8, drop_last=True)

        print (f'number of images is {len(image_dataset)}')

        class_idx= image_dataset.class_to_idx

        print (f'Classes with index are: {class_idx}')

        return data_loaders, image_dataset