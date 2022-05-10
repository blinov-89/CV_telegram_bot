from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import json

# load model
filepath = 'model/model_1.h5'
model = load_model(filepath, compile=True)

# parameter for loss function
smooth = 1.


#  metric function and loss function
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# load model_s
filepath_s = 'model/model_s.h5'
model_s = load_model(filepath_s, compile=True,
                     custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

def start(updater, context):
    updater.message.reply_text("Добро пожаловать в телеграм бот по классификации изображений")

def help_(updater, context):
    updater.message.reply_text("Отправь изображение")

def message(updater, context):
    msg = updater.message.text
    print(msg)
    updater.message.reply_text(msg)


def image(updater, context):
    photo = updater.message.photo[-1].get_file()
    updater.message.reply_text('Секундочку, я думаю...')

    photo.download("img.jpg")

    img = cv2.imread("img.jpg")
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (224, 224, 3))

    #  Преобразование изображения
    numpy_image = np.asarray(img)

    # прогноз классификации
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]

    # Opening JSON file
    label = open('labels.json', encoding="UTF-8")
    data = json.load(label)

    label.close()

    if np.amax(prediction) < 0.55:
        likely_class_pr = "Класс не определен, плохое качество фото, сфоткайте получше."
    else:
        likely_class_pr = data[str(np.argmax(prediction))]

    updater.message.reply_text('Это - {0}'.format(likely_class_pr))
    updater.message.reply_text('Точность: {p:.2f}%'.format(p=max(prediction.tolist()) * 100))
    updater.message.reply_text('Подождите ещё же маски')

    #  Преобразование изображения
    img = np.array(img) / 255.0

    # прогноз
    predict = np.zeros((224, 224, 3))
    #
    for i in range(3):
        predict_ = model_s.predict(np.expand_dims(img, axis=0))[i]
        predict_ = np.squeeze(predict_, axis=0)
        predict_ = np.squeeze(predict_, axis=2)
        predict[:, :, i] = predict_

    plt.imshow(predict)
    #
    # plt.savefig('predictim.png')
    #
    # photo = open('predictim.png', 'rb')
    #
    # updater.message.reply_photo(photo=photo)

    def draw_mask(img, mask):
      img = img * 255
      for k in range(3):
        img[:, :, k] += mask[:, :, k] * 255 * 0.5
      img = img.astype(np.uint8)
      plt.imshow(img)

    draw_mask(img, predict)

    plt.savefig('draw_mask.png')

    draw_mask = open('draw_mask.png', 'rb')

    updater.message.reply_photo(photo=draw_mask)


with open("TOKEN.txt", "r") as valuesFile:
    TOKEN = valuesFile.readlines()


updater = Updater(TOKEN[0])
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_))

dispatcher.add_handler(MessageHandler(Filters.text, message))
dispatcher.add_handler(MessageHandler(Filters.photo, image))

updater.start_polling()
updater.idle()
