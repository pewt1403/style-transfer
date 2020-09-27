import tkinter as tk
from tkinter import filedialog, ttk, StringVar
from PIL import ImageTk, Image, ExifTags
from PIL.ExifTags import TAGS
import os
import _thread
# TENSORFLOW
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import functools

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1400)])
    except RuntimeError as e:
        print(e)
# Path to picture
imgOriginal = ""
imgOriginal_W = 320
imgOriginal_H = 240
imgStyle = ""
imgStyle_W = 320
imgStyle_H = 240

imageOut = ''
imgArr = []
currentImg = 0

############# Progress bar logic ###############
progressValue = 0
def incProgValue():
    global progressValue
    p_loading['value'] = progressValue
    

################################################

############# Thread #############
def doMerge():
    _thread.start_new_thread( merge, ())

##################################

##################### AI ########################
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor )> 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
def merge():
    global progressValue
    global imgArr
    progressValue = 0
    if imgOriginal == "" or imgStyle == "":
        # imgpath = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save file", defaultextension=".jpg", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
        # print(imgpath)
        return
    
    b_imgOriginal["state"] = "disable"
    b_doMerge["state"] = "disable"
    b_imgStyle["state"] = "disable"
    b_exportBtn["state"] = "disable"

    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    content_image = load_img(imgOriginal)
    style_image = load_img(imgStyle)

    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def vgg_layers(layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    # Calculate style
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg =  vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs*255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
            style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
            content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
            style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
            return {'content':content_dict, 'style':style_dict}

    extractor = StyleContentModel(style_layers, content_layers)

    # Run gradient descent
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4
    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    # Train
    total_variation_weight=30

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    # start = time.time()
    epochs = 20
    steps_per_epoch = 50
    step = 0
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            progressValue += 100 / (epochs * steps_per_epoch)
            incProgValue()
            step += 1
            train_step(image)
            # print(".", end='')
        copyArr = image[:]
        imgArr.append(copyArr)
        # print(imgArr)
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        # print("Train step: {}".format(step))
        
    # end = time.time()
    # print("Total time: {:.1f}".format(end-start))

    # global imageOut
    # imageOut = tensor_to_image(image)
    # w, h = imageOut.size
    img = tensor_to_image(imgArr[0])
    w, h = img.size
    scale = h / w
    global i_imgOut
    i_imgOut = ImageTk.PhotoImage(img.resize((imgOriginal_W, int(round(imgOriginal_W * scale))), Image.ANTIALIAS))
    c_imgOut.create_image(10, 10, anchor="nw", image=i_imgOut)
    
    global picNo
    picNo.set("50 Iterations")
    b_imgOriginal["state"] = "normal"
    b_doMerge["state"] = "normal"
    b_imgStyle["state"] = "normal"
    b_exportBtn["state"] = "normal"
    b_exportAllBtn["state"] = "normal"
    b_prevImg['state'] = 'disable'
    b_nextImg['state'] = 'normal'
###############################################

#################### GUI #######################
root = tk.Tk(className='Style transfer')
root.geometry("800x640")
root.resizable(0, 0)
header = tk.Frame(root)
header.pack(side="top", fill="x")
inputImg = tk.Frame(root)
inputImg.pack(side="top", fill="x")
inputBtn = tk.Frame(root)
inputBtn.pack(side="top", fill="x")
outputImg = tk.Frame(root)
outputImg.pack(side="top", fill="x")
exportBtn = tk.Frame(root)
exportBtn.pack(side="top", fill="x")

picNo = StringVar()
picNo.set('')

# FUNCTION
def selectImg(imgType):
    imgpath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select " + imgType + " file", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    #print(imgpath)
    if(imgType == 'original' and imgpath != ''):
        global imgOriginal
        imgOriginal = imgpath
        image = Image.open(imgpath)
        w, h = image.size
        scale = h / w
        global i_imgOriginal
        i_imgOriginal = ImageTk.PhotoImage(image.resize((imgOriginal_W, int(round(imgOriginal_W * scale))), Image.ANTIALIAS))
        c_imgOriginal.create_image(10, 10, anchor="nw", image=i_imgOriginal)
    elif(imgType == 'style' and imgpath != ''):
        global imgStyle
        imgStyle = imgpath
        image = Image.open(imgpath)
        w, h = image.size
        scale = h / w
        global i_imgStyle
        i_imgStyle = ImageTk.PhotoImage(image.resize((imgStyle_W, int(round(imgStyle_W * scale))), Image.ANTIALIAS))
        c_imgStyle.create_image(10, 10, anchor="nw", image=i_imgStyle)
    return

def saveImg(amount):
    global imgArr
    global currentImg
    if(amount == 'all'):
        imgpath = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save all files", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        for i in range(len(imgArr)):
            num = (i + 1) * 50
            img = tensor_to_image(imgArr[i])
            img.save(imgpath + "_" + str(num) + ".jpg", dpi=(1200, 1200))
    elif(amount == 'one'):
        imgpath = filedialog.asksaveasfilename(initialdir=os.getcwd(), defaultextension='.jpg', title="Save file", confirmoverwrite=True, filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        img = tensor_to_image(imgArr[currentImg])
        img.save(imgpath, dpi=(1200, 1200))

def changeImg(state):
    global imgArr
    global currentImg
    global i_imgOut
    global picNo
    global l_picNo
    if state == 'next':
        currentImg += 1
        b_prevImg['state'] = 'normal'
        if currentImg == (len(imgArr) - 1):
            b_nextImg['state'] = 'disable'
        img = tensor_to_image(imgArr[currentImg])
        w, h = img.size
        scale = h / w
        i_imgOut = ImageTk.PhotoImage(img.resize((imgOriginal_W, int(round(imgOriginal_W * scale))), Image.ANTIALIAS))
        c_imgOut.create_image(10, 10, anchor="nw", image=i_imgOut)
    elif state == 'prev':
        currentImg -= 1
        b_nextImg['state'] = 'normal'
        if currentImg == 0:
            b_prevImg['state'] = 'disable'
        img = tensor_to_image(imgArr[currentImg])
        w, h = img.size
        scale = h / w
        i_imgOut = ImageTk.PhotoImage(img.resize((imgOriginal_W, int(round(imgOriginal_W * scale))), Image.ANTIALIAS))
        c_imgOut.create_image(10, 10, anchor="nw", image=i_imgOut)
    picNo.set(str(50 * (currentImg + 1)) + "Iterations")

# WIDGET
p_loading = ttk.Progressbar(header, orient="horizontal", length=800)

b_imgOriginal = tk.Button(inputBtn,width="20", height="2", text="Select Original Picture", command=lambda:selectImg('original'))
c_imgOriginal = tk.Canvas(inputImg, width=imgOriginal_W, height=imgOriginal_H)
# ImageTk.PhotoImage(Image.open(imgOriginal).resize((imgOriginal_W, imgOriginal_H), Image.ANTIALIAS))
i_imgOriginal = ''

b_imgStyle = tk.Button(inputBtn, width="20", height="2", text="Select Style Picture", command=lambda:selectImg('style'))
c_imgStyle = tk.Canvas(inputImg, width=imgStyle_W, height=imgStyle_H)
# ImageTk.PhotoImage(Image.open(imgStyle).resize((imgStyle_W, imgStyle_H), Image.ANTIALIAS))
i_imgStyle = ''

c_imgOut = tk.Canvas(outputImg, width=imgOriginal_W, height=imgOriginal_H)
#i_imgOut = ImageTk.PhotoImage(Image.open("a.jpg").resize((imgStyle_W, imgStyle_H), Image.ANTIALIAS))
b_nextImg = tk.Button(exportBtn,width="20",height="3", text="Next", command=lambda:changeImg('next'))
b_prevImg = tk.Button(exportBtn,width="20",height="3", text="Prev", command=lambda:changeImg('prev'))

b_doMerge = tk.Button(inputBtn,width="20",height="3", text="Merge!", command=doMerge)
b_exportBtn = tk.Button(exportBtn, width="20", height="3", text="Save image", command=lambda:saveImg('one'))
b_exportAllBtn = tk.Button(exportBtn, width="20", height="3", text="Save all images", command=lambda:saveImg('all'))
l_picNo = tk.Label(outputImg, textvariable=picNo)

p_loading.pack()

c_imgOriginal.pack(side="left")
#c_imgOriginal.create_image(10, 10, anchor="nw", image=i_imgOriginal)
c_imgStyle.pack(side="right")
#c_imgStyle.create_image(10, 10, anchor="nw", image=i_imgStyle)
#c_imgOut.create_image(10, 10, anchor="nw", image=i_imgOut)
c_imgOut.pack()

b_prevImg.pack(side='left')
b_nextImg.pack(side='right')
b_imgOriginal.pack(side='left', padx=(85,0))
b_doMerge.pack(side='left', padx=(95,0))
b_imgStyle.pack(side='right', padx=(0,85))
b_exportBtn.pack(side='left', padx=(85,0))
b_exportAllBtn.pack(side='right', padx=(0,85))

b_nextImg['state'] = 'disable'
b_prevImg['state'] = 'disable'
b_exportBtn['state'] = 'disable'
b_exportAllBtn['state'] = 'disable'

l_picNo.pack(side="bottom")

root.mainloop()