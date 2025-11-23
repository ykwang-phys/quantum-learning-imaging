#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:07:39 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:46:03 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:24:04 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:01:13 2025

@author: yunkaiwang
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig,eigh,inv
import matplotlib.pyplot as plt
import random
import time
from scipy import integrate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator, LogLocator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

# Load dataset
faces = fetch_olivetti_faces()
images = faces.images   # shape (400, 64, 64)
targets = faces.target  # labels 0–39

person=1
# Pick which face you want, e.g. face #10
idx = 0 + 10* person

plt.figure(figsize=(4,4))
plt.imshow(images[idx], cmap="gray")
#plt.title(f"Face #{idx}, person {targets[idx]}", fontsize=16)
plt.axis("off")
plt.savefig("olivetti_face_0.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()

idx = 1 + 10* person

plt.figure(figsize=(4,4))
plt.imshow(images[idx], cmap="gray")
#plt.title(f"Face #{idx}, person {targets[idx]}", fontsize=16)
plt.axis("off")
plt.savefig("olivetti_face_1.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()

idx =2 + 10* person

plt.figure(figsize=(4,4))
plt.imshow(images[idx], cmap="gray")
#plt.title(f"Face #{idx}, person {targets[idx]}", fontsize=16)
plt.axis("off")
plt.savefig("olivetti_face_2.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()

idx = 3 + 10* person

plt.figure(figsize=(4,4))
plt.imshow(images[idx], cmap="gray")
#plt.title(f"Face #{idx}, person {targets[idx]}", fontsize=16)
plt.axis("off")
plt.savefig("olivetti_face_3.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()



L=10
sigma=1
alpha=0.1
source_size=alpha*sigma
M=3
K=1000
y0 = np.array([-0.25,-0.05, 0.2]) * L
    
    
source_position=np.zeros(M*K)

y = np.linspace(-0.5, 0.5, M*K )*L
for j  in range(M*K):
    for q in range(M):
        if abs(y[j] - y0[q]) < source_size/2:
            source_position[j]=1
            
#plt.figure(figsize=(24,4))
plt.figure()
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.plot(y/L,source_position, linewidth=3)
#plt.title('source position', fontsize=25)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.5))
# make axis lines thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
#plt.tight_layout()
plt.ylim(-0.1,1.19)
plt.savefig('source_position.png', dpi=300, bbox_inches='tight')
plt.show()
    




