from PIL import Image
import numpy as np
import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def Random_crop(img, img_size):
    width, height = img.size
    width_range = width - img_size
    height_range = height - img_size
    random_ws = np.random.randint(width_range)
    random_hs = np.random.randint(height_range)
    random_wd = img_size + random_ws 
    random_hd = img_size + random_hs
    
    img = img.crop((random_ws, random_hs, random_wd, random_hd))
    
    return img


def Resize(img, img_size):
    return(img.resize((img_size, img_size),Image.ANTIALIAS))


def normalize(img, mean, std):
    img = img / 255.0
    mean=np.array(mean)
    mean = mean[:, None, None]
    std=np.array(std)
    std = std[:, None, None]
    img -= mean
    img /= std
    return img


def RandomHorizontalFlip(img):
    random_flag = np.random.randint(2)
    if random_flag:
        img=img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
    

class DatasetFolder(object):
    def __init__(self, root, loader, extensions, img_size, batch_size):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.niter=0
        self.length=len(self.samples)
        self.img_size = img_size
        self.batch_size = batch_size
        self.batch = []

    
    def __next__(self):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.niter < self.length:
            self.batch = []
            for i in range(self.batch_size):
                path, target = self.samples[self.niter]
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = Resize(img, self.img_size + 30)
                img = Random_crop(img, self.img_size)
                img=RandomHorizontalFlip(img)
                img=np.array(img)
                img=img.transpose((2, 0, 1))
                img=normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                self.niter += 1
                self.batch.append(img)
            return np.array(self.batch).astype('float32'), target  
        else:
            self.niter = 0
            self.batch = []
            
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Transforms: Normalized, RandomCrop, RandomHorizontalFlip.'
        
        return fmt_str


class ImageFolder(DatasetFolder):
    def __init__(self, root, img_size, batch_size, loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS, img_size, batch_size)