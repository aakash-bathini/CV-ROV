

# in case it matters: licensed under GPLv2 or later
# legend:
# sqr(x)  = x*x
# sqrt(x) = square root of x

# let m1x,m1y ... m4x,m4y be the (x,y) pixel coordinates
# of the 4 corners of the detected quadrangle
# i.e. (m1x, m1y) are the cordinates of the first corner, 
# (m2x, m2y) of the second corner and so on.
# let u0, v0 be the pixel coordinates of the principal point of the image
# for a normal camera this will be the center of the image, 
# i.e. u0=IMAGEWIDTH/2; v0 =IMAGEHEIGHT/2
# This assumption does not hold if the image has been cropped asymmetrically

# first, transform the image so the principal point is at (0,0)
# this makes the following equations much easier
import math
def sqrt(x):
    return math.sqrt(x)
    
def sqr(x):
    return x**2

u0 = 1920 / 2
v0 = 1080 / 2

m1x = 1110 - u0;
m1y = 918 - v0;
m2x = 1701 - u0;
m2y = 481 - v0;
m3x = 333 - u0;
m3y = 316 - v0;
m4x = 909 - u0;
m4y = 70 - v0;


# temporary variables k2, k3
k2 = ((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x) / ((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) 

k3 = ((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x) / ((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x)

# f_squared is the focal length of the camera, squared
# if k2==1 OR k3==1 then this equation is not solvable
# if the focal length is known, then this equation is not needed
# in that case assign f_squared= sqr(focal_length)
f_squared = -((k3*m3y - m1y)*(k2*m2y - m1y) + (k3*m3x - m1x)*(k2*m2x - m1x)) / ((k3 - 1)*(k2 - 1))

#The width/height ratio of the original rectangle
whRatio = sqrt( 
    (sqr(k2 - 1) + sqr(k2*m2y - m1y)/f_squared + sqr(k2*m2x - m1x)/f_squared) /
    (sqr(k3 - 1) + sqr(k3*m3y - m1y)/f_squared + sqr(k3*m3x - m1x)/f_squared) 
)

# if k2==1 AND k3==1, then the focal length equation is not solvable 
# but the focal length is not needed to calculate the ratio.
# I am still trying to figure out under which circumstances k2 and k3 become 1
# but it seems to be when the rectangle is not distorted by perspective, 
# i.e. viewed straight on. Then the equation is obvious:
if (k2==1 and k3==1):
    whRatio = sqrt( 
    (sqr(m2y-m1y) + sqr(m2x-m1x)) / 
    (sqr(m3y-m1y) + sqr(m3x-m1x)))


# After testing, I found that the above equations 
# actually give the height/width ratio of the rectangle, 
# not the width/height ratio. 
# If someone can find the error that caused this, 
# I would be most grateful.
# until then:
whRatio = 1/whRatio;

print(whRatio)