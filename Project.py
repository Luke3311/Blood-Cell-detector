#Import libraries
import numpy as np 
import cv2  as cv
import matplotlib.pyplot as plt


def cell_type(contour):
    #Function to differentiate abnormal cells from normal rbc
    
    #Global is used as we want to change global variables when running function
    global rbc_count, rbc,abnormal_cell , ac_count
    
    perimeter = cv.arcLength(contour,True)#Obtain perimeter of cell contour
                                          # True = contour is closed
                                          
    form_factor = 4*np.pi*a/(perimeter**2)#Form factor equation
    
    #Considered normal cell if ff is over 0.5 
    if form_factor > 0.5:
        rbc.append(contour)
        rbc_count +=1
    else:
        abnormal_cell.append(contour)
        ac_count +=1


# Read input image
input_img = cv.imread('SC4.jpg')


# Convert it to HSV Color Map
hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)


# Set the thresholds for RBC Detection
rbc_lower = np.array([150, 45, 40])
rbc_upper = np.array([180, 200, 255])

# Set the thresholds for WBC Detection
wbc_lower = np.array([100, 100, 20])
wbc_upper = np.array([150, 230, 255])


# Find the colors within the specified range and apply the mask
rmask = cv.inRange(hsv_img, rbc_lower, rbc_upper)
wmask = cv.inRange(hsv_img, wbc_lower, wbc_upper)


# Find the contours
rcontours, hierarchy = cv.findContours(rmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
wcontours, hierarchy = cv.findContours(wmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


# Count the number of RBC's
rbc_count = 0
wbc_count = 0
ac_count = 0

#Array for bc contours
rbc = []
wbc = [] 
abnormal_cell = []

#RBC typically overlap, so an array for multiple RBC is created
#To segment them when this happens
multiple_rcells = []
       


#Checking WBC contours and removing outliers
for w_contour in wcontours:
    b = cv.contourArea(w_contour)
    
    #All WBC usually fall within these areas, no need to obtain median
    if 470 < b < 4000:
        wbc.append(w_contour)
        wbc_count += 1

#temporary array usex for sorting
temp = []

#Obtain contours with areas > 70 and store their index and area
for i , area in enumerate(rcontours):
    a = cv.contourArea(area)
    if a > 70:
         temp.append([i,a])
         
#Calculate median rbc area
w = np.median([j[1] for j in temp])

for c in temp:
    a = c[1]
    
    if a > (w-(w*1/2)) and a <= (w+(w*1/2)): #Estimated 1 RBC lies within this contour
        cell_type(rcontours[c[0]])           #RBC is checked to see if it is abnormal
        
    elif a > (w+(w*1/2)) and a <= 2.5*w:     #Estimated 2 RBC lies within this contour
        rbc_count +=2
        multiple_rcells.append(rcontours[c[0]])
        
    elif a > 2.5*w:                         #Estimated 3 RBC lies within this contour
        rbc_count +=3
        multiple_rcells.append(rcontours[c[0]])


     
        
# Count and plot the RBC
cv.drawContours(input_img, multiple_rcells, -1, (0, 0, 255), 3)
cv.drawContours(input_img, rbc, -1, (0, 0, 255), 3)

# Count and plot the WBC
cv.drawContours(input_img, wbc, -1, (255, 0, 0), 3)

#Count and plot the abnormnal RBC
cv.drawContours(input_img, abnormal_cell , -1, (0, 255, 0), 3)


# Print the counts
print(f'Estimate number of red blood cells is: {rbc_count}')
print(f'Estimate number of white blood cells is: {wbc_count}')
print(f'Estimate number of abnormal red blood cells is: {ac_count}')


# Write the output in a file
cv.imwrite('detection4.jpg', input_img)


