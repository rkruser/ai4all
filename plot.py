# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:08:45 2019

@author: Student
"""
import pickle

import matplotlib.pyplot as plt

f = open('./models/your_network_savedvalues.pkl', 'rb')
savedvalues = pickle.load(f)
train_accs, train_losses, val_accs, val_losses = savedvalues
print(type(train_accs), train_accs)
#print(savedvalues)

#train pochs, test epochs

plt.plot(train_accs)
plt.plot(val_accs)

plt.xlabel("training epochs")
plt.ylabel("accuracy")


#plt.legend(('Linear', 'Logistic'),
           #loc="lower right", fontsize='small')
           
# Show the plot
#plt.show()