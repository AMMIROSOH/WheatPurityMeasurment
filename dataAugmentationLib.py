import os
import cv2 as cv
import random

def readImagesFromDirectories(*directories):
    images = []
    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory path.")
            continue
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(directory, filename)
                img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Error: Unable to read image {image_path}.")
    return images

def overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def placeImage(image, mask, canvas, objects, seed=True):
    imgHeight, imgWidth = image.shape[:2]
    thresholdSeed = createMaskSeed(image, seed)
    maxAttempts = 100
    for _ in range(maxAttempts):
        x = random.randint(0, canvas.shape[1] - imgWidth)
        y = random.randint(0, canvas.shape[0] - imgHeight)
        if all(not overlap((x, y, imgWidth, imgHeight), (obj['x'], obj['y'], obj['w'], obj['h'])) for obj in objects):
            alpha = image[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+imgHeight, x:x+imgWidth, c] = (1.0 - alpha) * canvas[y:y+imgHeight, x:x+imgWidth, c] + alpha * image[:, :, c]

            mask[y:y+imgHeight, x:x+imgWidth] = thresholdSeed
            objects.append({'x': x, 'y': y, 'w': imgWidth, 'h': imgHeight})
            return True
    return False

def createMaskSeed(image, seed=True):
    hsiImage = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
    saturationComponent = hsiImage[:, :, 1]
    if seed:
        _, thresholdImage = cv.threshold(saturationComponent, 30, 255, cv.THRESH_BINARY)
    else: 
        _, thresholdImage = cv.threshold(saturationComponent, 30, 127, cv.THRESH_BINARY) # 127
    return thresholdImage

def placeImagesOnBackground(backgroundImg, wheatImages, miscImages, seedChanceRef=0.3):
    canvas = backgroundImg.copy()
    mask = backgroundImg[:, :, 1].copy()
    mask[:, :] = 0
    
    objects = []
    for i in range(random.randint(300, 800)):  # len(wheatImages) + len(miscImages)
        seedChance = random.randint(0, 1)
        if seedChance >= seedChanceRef:
            placeImage(random.choice(wheatImages), mask, canvas, objects)
        else:
            placeImage(random.choice(miscImages), mask, canvas, objects, False)

    return canvas, mask
