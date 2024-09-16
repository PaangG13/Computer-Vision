import json
from pycocotools.coco import COCO
json_path="./data/train.json"
cocoGt=COCO(json_path)
categories = cocoGt.loadCats(cocoGt.getCatIds())
#print(categories)
categoryNames = [cat['name'] for cat in categories]
#print(categoryNames[0])
#print('Category Name List：', categoryNames)
imageCount = len(cocoGt.imgs)
print('Number of images：', imageCount)
i=0
for catId in cocoGt.getCatIds():
    print("Category: {}, Number: {}".format(categoryNames[i], len(cocoGt.catToImgs[catId])))
    i+=1


# coco = COCO(json_path)
# image_ids = set()
# for ann_id in coco.getAnnIds():
#     ann = coco.loadAnns([ann_id])[0]
#     image_id = ann['image_id']
#     if image_id in image_ids:
#         print('The image with ID {} contains two or more object detection targets.'.format(image_id))
#     else:
#         image_ids.add(image_id)
