import numpy as np
x = np.linspace(0, 5, 11) # Create 11 evenly spaced values between 0 and 5 (including both 0 and 5)
y = x ** 2  # Compute the square of each value in x (element-wise), i.e., y = x²

#plt.figure(figsize=(2,2))         # Set display figure size
plt.plot(x, y, 'r')               # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()



# plt.subplot(nrows, ncols, plot_number)
plt.subplot(1,2,1)
plt.plot(x, y, 'r--') # More on color options later

plt.subplot(1,2,2)
plt.plot(y, x, 'g*-');




# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(x, y, 'b')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')



# Creates blank canvas
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes1')
axes1.set_ylabel('Y_label_axes1')
axes1.set_title('Axes 1 Title')

# Insert Figure Axes 2
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Axes 2 Title');



# Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.subplots()

# Now use the axes object to add stuff to plot
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');



# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)


# Axes is an array of axes to plot on
axes

#Array Iteration
for ax in axes:
    ax.plot(x, y, 'b')  #Plot On That Set Of Axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

# Display the figure object    
fig


#Layout Manager Using subplots Function For More Automatic Axis Manager
fig, axes = plt.subplots(nrows=1, ncols=2) 

for ax in axes:
    ax.plot(x, y, 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

fig    
#Automatically Adjusts The Position Of The Axes On the Figure Canvas So There's No Overlapping Content
plt.tight_layout() 


#Specifies The Aspect Ratio And Dots-Per-Inch (DPI) Upon Figure Object Creation
fig = plt.figure(figsize=(8,4), dpi=100)



#To Save A Figure To A File
fig.savefig("filename.png")




#Save A Figure With Chosen DPI Value
fig.savefig("filename.png", dpi=200)




#To Add A Title In Each Axis
ax.set_title("title");



#Axis Label For Each Dimension
ax.set_xlabel("x")
ax.set_ylabel("y");




#Create Figure With Empty Canvas
fig = plt.figure() 
#Add Set Of Axes To Figure
ax = fig.add_axes([0,0,1,1]) # [a, b, c, d], where a is left, b is bottom, c is width, d is height (0 to 1)
#Plot On That Set Of Axes
ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend() #To Add The Legend To The Figure






#Creates a 1x3 grid of subplots with a figure of 12x4 
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#Plot on first set of axes
axes[0].plot(x, x**2, x, x**3) #Plot On That Set Of Axes
axes[0].set_title("default axes ranges")

#Plot on second set of axes with tight axes range
axes[1].plot(x, x**2, x, x**3) #Plot On That Set Of Axes
axes[1].axis('tight') #'tight' To Automatically Get Tightly Fitted Axes Ranges
axes[1].set_title("tight axes")

#Plot on third set of axes with custom axis ranges
axes[2].plot(x, x**2, x, x**3) #Plot On That Set Of Axes
axes[2].set_ylim([0, 60])
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range");


    
#SEABORN


