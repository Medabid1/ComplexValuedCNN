# ComplexValuedCNN

Complex CNN in Pytorch, implementation of the paper Deep Complex Networks : https://arxiv.org/abs/1705.09792


### Modules implemeted : 
-[x] Complex Conv2d
-[x] Complex ConvTranspose2d
-[x] Complex AvgPool2d
-[x] Complex Linear

### Activation function : 
-[x] zRelu
-[x] CRelu

## How to use :
Define your input using the class CPLX by passing the real and imaginary part to it.
    x = CPLX(real_part, imaginary_part)
