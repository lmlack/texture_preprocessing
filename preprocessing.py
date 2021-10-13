#!/usr/bin/env python
# coding: utf-8

# Creates train, validation (if specified), and test sets.  Input is a training folder of images and a test folder of images, both of which get tiled as saved as .npz files.
# Assumption: input image dimensions are square with height (=width) that is a multiple of SCALE*TILESIZE

# Tweakable/constant parameters
# Note: Modified images stored in temp directory

#Note: Change to not accept more than 16 tiles if image is larger than 1024x1024

# ---Note to self: Change this later - have better separation of test and train datasets---
TRAIN_INPUT_DIR =       '/content/gdrive/MyDrive/Copy_Paper_Images'   # directory of input images in training set
TRAIN_OUTPUT_TILE_DIR = '/tmp/data'    # temp directory where tiled training images go
TEST_INPUT_DIR =        '/content/gdrive/MyDrive/Copy_Paper_Images'    # directory of input images in test set
TEST_OUTPUT_TILE_DIR =  '/tmp/data'      # temp directory where tiled test images go
TRAIN_AUGMENTEDTILES = 16                   # number of augmented tiles to put in training/validation sets
TRAIN_VALIDATIONTILES = 3                   # number of tiles (drawn from regular+augmented) to use for validation
SCALE = 4                                   # image downsampling factor in creating tiles
TILESIZE = 64                               # size of each tile after image downsampling (determines # tiles/img)
MAX_TILES = 16

def make_dataset(input_dir, output_dir, SCALE, TILESIZE, augmentedtiles=0, validationtiles=0):
    # function inputs:
    #    input_dir -- directory of input images
    #    output_dir -- directory of tiled images that will get created
    #    SCALE -- downsampling factor
    #    TILESIZE -- number of pixels in each edge of (square) tile that gets created
    #    augmentedtiles -- number of augmented tiles to create
    #    validationtiles -- number of tiles per class to put in validation set

    if os.path.exists(output_dir):  # if specified tile directory exists, wipe it and start fresh
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    imgs = os.listdir(input_dir)
    #imgs = [img for img in imgs if not 'zoomout' in img]
    for i in range(len(imgs)):                         # loop through each image
        pic_name = imgs[i]                             # get image name
        img = Image.open(input_dir + '/' + pic_name)                    # load image
        img = img.resize((img.size[0]//SCALE,img.size[1]//SCALE))       # downsample image by SCALE factor
        dir_name = output_dir + '/' + os.path.splitext(pic_name)[0]     # directory name for tiles from this image
        os.mkdir(dir_name)                                              # create directory
        digits = math.floor(math.log((img.size[0]//TILESIZE)**2,10))+1  # number of digits needed to represent all tiles
        tilecounter = 0                                                 # tile counter for file name of tile images

        # loop over all tiles, crop and save tile, increment counter
        for xx in range(0,img.size[0],TILESIZE):
            for yy in range(0,img.size[0],TILESIZE):
              if tilecounter < MAX_TILES:
                  img.crop((xx,yy,xx+TILESIZE,yy+TILESIZE)).save(dir_name + '/' + str(tilecounter).zfill(digits) + '.png')
                  tilecounter += 1

        # grab more randomly sampled tiles from image, with random brightness and contrast
        for j in range(0,augmentedtiles):
            xx = random.randrange(img.size[0]-TILESIZE+1)       # random x coord
            yy = random.randrange(img.size[0]-TILESIZE+1)       # random y coord
            tile = img.crop((xx,yy,xx+TILESIZE,yy+TILESIZE))    # random crop selection from image
            tile = ImageEnhance.Brightness(tile).enhance(random.random()+0.5)  # random brightness adjustment between 0.5 --> 1.5
            tile = ImageEnhance.Contrast(tile).enhance(random.random()+0.5)    # random contrast adjustment between 0.5 --> 1.5
            tile.save(dir_name + '/' + str(tilecounter).zfill(digits) + '.png')
            tilecounter += 1

    # use tensorflow's flow_from_directory to load/create dataset
    train_datagen = ImageDataGenerator(rescale = 1./255)
    batch_size = sum([len(files) for r, d, files in os.walk(output_dir)])
    photopaper_gen = train_datagen.flow_from_directory(
        directory = output_dir,
        target_size = (TILESIZE, TILESIZE),
        color_mode = 'grayscale',
        shuffle=False,
        batch_size = batch_size)

    # convert to numpy array and save
    x = photopaper_gen[0][0].squeeze()                        # input data
    y = [np.where(r==1)[0][0] for r in photopaper_gen[0][1]]  # integer version instead of one-hot
    fnames = photopaper_gen.filenames                         # save filenames
    classnames = [r[0] for r in list(photopaper_gen.class_indices.items())]  # save class names

    # partition into validation set
    tiles_per_image = tilecounter
    num_images = len(classnames)
    idx_train=np.zeros((tiles_per_image-validationtiles)*num_images, dtype=int)
    idx_valid=np.zeros(validationtiles*num_images, dtype=int)
    for i in range(num_images):
        idx = np.asarray(random.sample(range(tiles_per_image), tiles_per_image))+i*tiles_per_image
        idx_train[i*(tiles_per_image-validationtiles):(i+1)*(tiles_per_image-validationtiles)] = idx[validationtiles:]
        idx_valid[i*validationtiles:(i+1)*validationtiles] = idx[:validationtiles]

    idx_train=np.sort(idx_train)
    idx_valid=np.sort(idx_valid)

    return [x[idx_train,:], np.array(y)[idx_train], np.array(fnames)[idx_train], np.array(classnames), x[idx_valid,:], np.array(y)[idx_valid]]

# create training (and possibly validation) set
[x_train, y_train, fnames_train, classnames_train, x_valid, y_valid] = make_dataset(TRAIN_INPUT_DIR, TRAIN_OUTPUT_TILE_DIR, SCALE, TILESIZE, TRAIN_AUGMENTEDTILES, TRAIN_VALIDATIONTILES)

# create test set
[x_test, y_test, fnames_test, classnames_test, _, _] = make_dataset(TEST_INPUT_DIR, TEST_OUTPUT_TILE_DIR, SCALE, TILESIZE)

x_train=x_train[:,:,:,np.newaxis] # add dimension
x_test=x_test[:,:,:,np.newaxis]

# Processes all images with CLAHE
def equalize(x):
  imgs = np.empty((len(x),64,64), dtype=np.uint8)
  clahe = cv2.createCLAHE(clipLimit = 5)
  for i in range(len(x)):
    #print(x_train[i].shape)

    imgs[i] = np.squeeze(255*x[i]).astype(np.uint8)

    imgs[i] = clahe.apply(imgs[i])
    x[i] = imgs[i][:,:,np.newaxis]/255
    #imgs[i] = clahe(imgs[i], 5) + 30

equalize(x_train)
equalize(x_test)
