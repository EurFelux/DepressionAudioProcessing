from CNN_torch import adapt_shape, load_model
import torch

model_path = './models/model_1.h5'

model = load_model(model_path)

x = torch.normal(0, 1, size=(20, 1, 20, 1))
y = model.predict(x, return_type='pt')
print(y)
