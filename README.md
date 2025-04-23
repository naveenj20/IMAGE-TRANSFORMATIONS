# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary libraries and read the original image and save it as a image variable.

### Step2:
Translate the image using a function warpPerpective()

### Step3:
Scale the image by multiplying the rows and columns with a float value.

### Step4:
Shear the image in both the rows and columns.

### Step5:
Find the reflection of the image.

### Step6 :
Rotate the image using angle function.

## Program:
```python
Developed By: Naveen Jaisanker
Register Number: 212224110039

i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image=cv2.imread("flower.jpg")
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M=np.float32([[1,0,50],  [0,1,100],  [0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()

ii) Image Scaling
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("flower.jpg")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M = np.float32([[1.5,0,0],[0,1.7,0],[0,0,1]])
scaled_img = cv2.warpPerspective(org_image,M,(cols*2,rows*2))
plt.imshow(org_image)
plt.show()

iii)Image shearing
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("flower.jpg")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M_Y = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
sheared_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols*1.5),int(rows*1.5)))
sheared_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols*1.5),int(rows*1.5)))
plt.imshow(sheared_img_xaxis)
plt.show()
plt.imshow(sheared_img_yaxis)
plt.show()

iv)Image Reflection
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("flower.jpg")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M_Y = np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
reflected_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols),int(rows)))
reflected_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols),int(rows)))
plt.imshow(reflected_img_xaxis)
plt.show()
plt.imshow(reflected_img_yaxis)
plt.show()

v)Image Rotation
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("flower.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))

plt.imshow(rotated_img)
plt.show()

vi)Image Cropping
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("flower.jpg")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
cropped_img=org_image[80:900,80:500]
plt.imshow(cropped_img)
plt.show()

```
## Output:

### i)Image Translation

![Screenshot 2025-04-23 132607](https://github.com/user-attachments/assets/e9e142d3-d218-43dc-8ca2-6750ad95ac9c)

### ii) Image Scaling

![Screenshot 2025-04-23 132613](https://github.com/user-attachments/assets/e33c67b8-be52-4787-933a-3ea5b60bcb94)

### iii)Image shearing

![Screenshot 2025-04-23 132622](https://github.com/user-attachments/assets/caf54470-6d63-484e-afba-2a1a39dbd1e1)

### iv)Image Reflection

![Screenshot 2025-04-23 132627](https://github.com/user-attachments/assets/6f2e0246-7e6e-4c61-81b5-65851773ed36)

### v)Image Rotation

![Screenshot 2025-04-23 132632](https://github.com/user-attachments/assets/734ab8b7-6ad5-4381-8782-72e60457d907)

### vi)Image Cropping

![Screenshot 2025-04-23 132637](https://github.com/user-attachments/assets/0a3cc05e-da44-4b0a-a69b-269383900bf1)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
