import skimage.io as io
import pylab
import time as time
import json
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            print(dataset)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']

        else:
            cats = self.dataset['categories']
            # print(' ')
            # print('keypoints的cat就只有人1种')
            # print(cats)
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
            # print(cats)
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            # 根据imgIds找到所有的ann
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            # 通过各类条件如catIds对anns进行筛选
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:

            ids = [ann['id'] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

                if 'keypoints' in ann and type(ann['keypoints']) == list:

                    # turn skeleton into zero-based index
                    sks = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]) - 1
                    # print(sks)
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            # 画点之间的连接线
                            plt.plot(x[sk], y[sk], linewidth=1, color=c)
                    # 画点
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=4, markerfacecolor=c, markeredgecolor='k',
                             markeredgewidth=1)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=4, markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=1)
            p = PatchCollection(polygons, facecolor=color, linewidths=3, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=12)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])


pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annFile = '../data/COCO2017/annotations/person_keypoints_val2017.json'
img_prefix = '../data/COCO2017/val2017'

# initialize COCO api for instance annotations
coco = COCO(annFile)

# getCatIds(catNms=[], supNms=[], catIds=[])
# 通过输入类别的名字、大类的名字或是种类的id，来筛选得到图片所属类别的id
catIds = coco.getCatIds(catNms=['person'])

# getImgIds(imgIds=[], catIds=[])
# 通过图片的id或是所属种类的id得到图片的id
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[1407])

# loadImgs(ids=[])
# 得到图片的id信息后，就可以用loadImgs得到图片的信息了
# 在这里我们随机选取之前list中的一张图片
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
print(img['file_name'])
I = io.imread('%s/%s' % (img_prefix, img['file_name']))
plt.imshow(I)
plt.axis('off')
ax = plt.gca()

# getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=None)
# 通过输入图片的id、类别的id、实例的面积、是否是人群来得到图片的注释id
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

# loadAnns(ids=[])
# 通过注释的id，得到注释的信息
anns = coco.loadAnns(annIds)
print('\n')
print("``````````````",anns)

coco.showAnns(anns)
plt.imshow(I)
plt.axis('off')
plt.show()