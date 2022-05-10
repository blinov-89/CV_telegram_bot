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

![image](https://user-images.githubusercontent.com/61515881/167557037-a4950eff-06ab-40fd-a45c-b36640b854a6.png)
