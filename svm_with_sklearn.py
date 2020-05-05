import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
X = recipes[['Flour','Sugar']]
ingredients = recipes[['Flour','Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', 0, 1)

print(ingredients)
print('----------')
print(type_label)
# type_label = []

# for idx in recipes['Type']:
#     if idx == 'Muffin':
#         type_label.append(0)
#     elif idx == 'Scone':
#         type_label.append(1)
#     else:
#         type_label.append(2)

recipe_features = recipes.columns.values[1:].tolist()

model = SVC(kernel='linear')
model.fit(ingredients, np.array(type_label))

# Get the separating hyperplane
w = model.coef_[0]
# print(w)
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]
# print(model.intercept_[0])
# print(model.intercept_)

# Plot the parallels to the separating hyperplane that pass through the support vectors
# b = model.support_vectors_[0]
# # print(b)
# yy_down = a * xx + (b[1] -
# a * b[0])
# b = model.support_vectors_[-1]
# print(model.support_vectors_)
# yy_up = a * xx + (b[1] - a * b[0])

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', fit_reg=False)
# plt.plot(xx, yy_up, linewidth=1, color='black',)
# plt.plot(xx, yy_down, linewidth=1, color='black')
plt.plot(xx, yy, linewidth=2, color='black')

plt.show()
#
plt.scatter(X['Flour'], X['Sugar'], c=type_label, s=30, cmap=plt.cm.Paired)
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
print(Z)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.xlabel('Flour')
plt.ylabel('Sugar')
# plt.legend()
plt.show()