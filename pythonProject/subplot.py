import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211) #First plot
plt.plot([4,6,1,8,3,2,5]) #plot
plt.subplot(212)    #second plot
plt.plot([10,1,15,2,4,6,7,20])#plot
plt.title("Subplot")
plt.show()