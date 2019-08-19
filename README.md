# ComplexValuedCNN

Complex CNN in Pytorch, implementation of the paper Deep Complex Networks : https://arxiv.org/abs/1705.09792


### Modules implemeted : 
- [x] Complex Conv2d
- [x] Complex ConvTranspose2d
- [x] Complex AvgPool2d
- [x] Complex Linear

### Activation function : 
- [x] zRelu
- [x] CRelu

## Usage :
1. Define your Complex input using the class CPLX by passing the real and imaginary part to it.
    'x = CPLX(real_part, imaginary_part)'
2. Build your model using src modules.
3. Use the loss function in 'utils.py' to train your model. if the labels are real valued, set 'use_magnitude = True'

#### ToDo
- [] Complex Unet


