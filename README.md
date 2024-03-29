# CV_telegram_bot
____
classification and segmentation image
____
### «КЛАССИФИКАЦИЯ И СЕГМЕНТАЦИЯ ОБЪЕКТОВ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТЕЙ»
____
Датасет с фотографиями товаров, ценников и ценой товара
15 классов товаров:

{
	"0": "чоко пай",
	"1": "ликер Куантро",
	"2": "кола 0,5л",
	"3": "кола 1л",
	"4": "эрмигут",
	"5": "фруто няня",
	"6": "гранатовый сок",
	"7": "гречка",
	"8": "хенесси",
	"9": "мороженое",
	"10": "пепси 0,5л",
	"11": "пепси 1л",
	"12": "водка Байкал",
	"13": "водка Калашников",
	"14": "водка Первак"
}

9000 изображений
____
### Архитектура EfficientNetB0
____
При классификации используется предобученная model EfficientNetB0 с весами imagenet, входные размеры изображений 224х224х3

Данные поделены на train, test, validation выборки:

- train = 6000
- test = 1500
- validation = 1500

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
              
- loss: 0.2158
- acc: 0.9297

Разметка данных для обучения модели

При разметка данных использовали VGG image annotation

Разметка:
- Товар
- Ценник
- Цена
Cохранение в разных форматах для загрузки данных

Архитектура модели в стиле U-Net Xception

Получение предсказания расположения товара, ценника и цены товара
____
Разработан Телеграм бот, пользователь отправляет фото, в ответ получает класс, точность предсказания и маски расположения товара, ценника, цены
____
<div class="img-div">
  <img src="https://user-images.githubusercontent.com/61515881/167557037-a4950eff-06ab-40fd-a45c-b36640b854a6.png" width="300" />
  <img src="https://user-images.githubusercontent.com/61515881/167557457-bcbcca24-673b-4824-a81d-13e1e2461a87.png" width="300" />
</div>

____

"CLASSIFICATION AND SEGMENTATION OF OBJECTS USING NEURAL NETWORKS"

____

Dataset with photos of goods, price tags and the price of goods of 15 classes of goods 9000 images

EfficientNetB0 Architecture

The classification uses a pre-trained model EfficientNetB0 with imagenet weights, the input image sizes are 224x224x3, the data is divided into train, test, validation samples: train = 6000 test = 1500 validation = 1500 model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc']) loss: 0.2158 acc: 0.9297

Data markup VGG image annotation Markup was used to train the model When marking up data: Product Price Tag Price Saving in different formats for data loading

The architecture of the model in the style of U-Net Xception Obtaining a prediction of the location of the goods, the price tag and the price of the goods

A telegram bot has been developed, the user sends a photo, in response receives a class, accuracy of prediction and masks of the location of the product, price tag, price
