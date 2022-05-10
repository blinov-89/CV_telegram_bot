# CV_talagram_bot

classification and segmentation image

«КЛАССИФИКАЦИЯ И СЕГМЕНТАЦИЯ ОБЪЕКТОВ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТЕЙ»

Датасет с фотографиями товаров, ценников и ценой товара
15 классов товаров
9000 изображений

Архитектура EfficientNetB0

При классификации используется предобученная model EfficientNetB0 с весами imagenet, входные размеры изображений 224х224х3
Данные поделены на train, test, validation выборки:
train = 6000
test = 1500
validation = 1500
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
loss: 0.2158
acc: 0.9297

Разметка данных для обучения модели
При разметка данных использовали VGG image annotation
Разметка:
Товар
Ценник
Цена
Cохранение в разных форматах для загрузки данных

Архитектура модели в стиле U-Net Xception
Получение предсказания расположения товара, ценника и цены товара

Разработан Телеграм бот, пользователь отправляет фото, в ответ получает класс, точность предсказания и маски расположения товара, ценника, цены

![image]()
<img src="https://user-images.githubusercontent.com/61515881/167557037-a4950eff-06ab-40fd-a45c-b36640b854a6.png" width="300" />
![image](https://user-images.githubusercontent.com/61515881/167557457-bcbcca24-673b-4824-a81d-13e1e2461a87.png)

"CLASSIFICATION AND SEGMENTATION OF OBJECTS USING NEURAL NETWORKS"

Dataset with photos of goods, price tags and the price of goods of 15 classes of goods 9000 images

EfficientNetB0 Architecture

The classification uses a pre-trained model EfficientNetB0 with imagenet weights, the input image sizes are 224x224x3, the data is divided into train, test, validation samples: train = 6000 test = 1500 validation = 1500 model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc']) loss: 0.2158 acc: 0.9297

Data markup VGG image annotation Markup was used to train the model When marking up data: Product Price Tag Price Saving in different formats for data loading

The architecture of the model in the style of U-Net Xception Obtaining a prediction of the location of the goods, the price tag and the price of the goods

A telegram bot has been developed, the user sends a photo, in response receives a class, accuracy of prediction and masks of the location of the product, price tag, price
