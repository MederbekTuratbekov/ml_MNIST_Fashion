from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
import streamlit as st
import torch.nn as nn
from PIL import Image
import uvicorn
import torch
import io



app = FastAPI()

class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('model_MnistFashion.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.post('/predict')
async def check_image(file: UploadFile = File()):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(400, detail='Файл кошулган жок')

        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            y_prediction = model(image_tensor)
            prediction = y_prediction.argmax(dim=1).item()
        return {
            'Класстын саны': prediction,
            'Класстын аталышы': class_names[prediction]
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# st.title('Mnist Fashion Model')
# st.text('Загрузите изображение одежды, и модель попробует её распознать.')
#
# mnist_image = st.file_uploader('Выберите изображение', type=['PNG', 'JPG', 'JPEG', 'SVG'])
#
# if not mnist_image:
#     st.info('Загрузите изображение')
# else:
#     st.image(mnist_image, caption='Загруженное изображение')
#
#     if st.button('Распознать одежду'):
#         try:
#             image = Image.open(mnist_image)
#             image_tensor = transform(image).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_prediction = model(image_tensor)
#                 prediction = y_prediction.argmax(dim=1).item()
#             st.success(f'Модель думает, что это: {class_names[prediction]}')
#
#         except Exception as e:
#             st.error(f'Ошибка: {str(e)}')
