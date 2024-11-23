from imutils import paths
import numpy as np
import shutil
import os
import pandas as pd



# specify path to the flowers and mnist dataset
DATASET_PATH = "/data/images"
# specify the paths to our training and validation set 
TRAIN = "/data/train"
VAL = "/data/val"
# set the input height and width
INPUT_HEIGHT = 450
INPUT_WIDTH = 600
# set the batch size and validation data split
BATCH_SIZE = 8
VAL_SPLIT = 0.1



def copy_images(Images, folder,DF):
    # check if the destination folder exists and if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    # loop over the image paths
    for image in Images:
        # grab image name and its label from the path and create
        # a placeholder corresponding to the separate label folder
        imageName = image
        
        label = DF[DF['image']==image]['label'].item()
        
        labelFolder = os.path.join(folder, label)
        # check to see if the label folder exists and if not create it
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
        # construct the destination image path and copy the current
        # image to it
        destination = os.path.join(labelFolder, imageName+'.jpg')
        shutil.copy('data/images/'+imageName+'.jpg', destination)
        







df = pd.read_csv('data/GroundTruth.csv')

result = df.apply(lambda row: row[row==1].index.tolist(),axis=1)
result = result.apply(lambda x: x[0] if x else None)
dict = {'image':df['image'],'label':result}
df_new = pd.DataFrame(dict)

images = list(df_new['image'])
np.random.shuffle(images)

# generate training and validation paths
valLen = int(len(images) * VAL_SPLIT)
trainsLen = len(images) - valLen
train = images[:trainsLen]
val = images[trainsLen:]
    


copy_images(train, TRAIN,df_new)
copy_images(val, VAL,df_new)















