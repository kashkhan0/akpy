#!/usr/bin/python
#   python veb262.py 212 ^C -12 400
#
#python directmbfm-1108.py 0004  38.34492110055494  -122.45438575744627 500
#
import sys, os, random, datetime, json
 
import string, re
from decimal import Decimal
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
import linecache
import math
import arrow
import numpy as np
import cv2
import oc
import requests
import struct
import base64,zlib
from array import array
from StringIO import StringIO
from shutil import copy2
import shutil
# import numexpr as ne
from colorama import Fore, Back, Style

 
sdlist = {"01": "457A-4E95", "02": "4638-7378", "05":"4571-6119","33":"4923-630D","34":"","35":"4923-6305","36": "4923-6311", "37": "4923-"}

 # time cp -rn sdd1/DCIM ssd500-b553/20160910/20160910-4923-630D/



"""
quadlist.py  list tiles under particlar quadkey
tileupsample.py generate higher zooms
oc.getimuritiles(imuri,20) get tiles for georeffed image


"""




earthRadius = 6378137
earthCircumference = earthRadius * 2 * math.pi
pixelsPerTile = 256
projectionOriginOffset = earthCircumference / 2
minLat = -85.0511287798
maxLat = 85.0511287798
PI = 3.1415926535897932384

CBK = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472]
CEK = [0.7111111111111111, 1.4222222222222223, 2.8444444444444446, 5.688888888888889, 11.377777777777778, 22.755555555555556, 45.51111111111111, 91.02222222222223, 182.04444444444445, 364.0888888888889, 728.1777777777778, 1456.3555555555556, 2912.711111111111, 5825.422222222222, 11650.844444444445, 23301.68888888889, 46603.37777777778, 93206.75555555556, 186413.51111111112, 372827.02222222224, 745654.0444444445, 1491308.088888889, 2982616.177777778, 5965232.355555556, 11930464.711111112, 23860929.422222223, 47721858.844444446, 95443717.68888889, 190887435.37777779, 381774870.75555557, 763549741.5111111]
CFK = [40.74366543152521, 81.48733086305042, 162.97466172610083, 325.94932345220167, 651.8986469044033, 1303.7972938088067, 2607.5945876176133, 5215.189175235227, 10430.378350470453, 20860.756700940907, 41721.51340188181, 83443.02680376363, 166886.05360752725, 333772.1072150545, 667544.214430109, 1335088.428860218, 2670176.857720436, 5340353.715440872, 10680707.430881744, 21361414.86176349, 42722829.72352698, 85445659.44705395, 170891318.8941079, 341782637.7882158, 683565275.5764316, 1367130551.1528633, 2734261102.3057265, 5468522204.611453, 10937044409.222906, 21874088818.445812, 43748177636.891624]





class Config(object):
    """ a class to hold constants"""
 
    FlightImagery = '/home/oc/F5T/oc/Flight-Imagery/'
    pylink="/home/oc/F5T/oc/Flight-Imagery/work/pylink/"
    veroot='/home/oc/F5T/oc/Flight-Imagery/work/ve/'
 
    black=(0,0,0)
    exe="./akk2" # fm executable

    #processed = requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/control.php?config=processed&if=REST").text

def system(cmd, debug = True):
  if debug:
    coutb( cmd)
  return os.system(cmd)
def tnow( text = "tt"):
  e = arrow.utcnow()
  return  e.timestamp + e.microsecond / 1e6

def thuman():
  e = arrow.utcnow().to('US/Pacific')
  return  e.format('YYYY-MM-DD HH.mm.ss ZZ')
def today():
  e = arrow.utcnow().to('US/Pacific')
  return  e.format('YYYY-MM-DD')

def tstr():
  e = arrow.utcnow().to('US/Pacific')
  return  e.format('YYYYMMDD-HHmmss')


def b64encode(a):
  return base64.b64encode(a)









def list2corn(p):
  return  np.float32([[ p[0][0],p[0][1] ],[p[1][0],p[1][1] ],[p[2][0],p[2][1] ],[p[3][0],p[3][1] ]]).reshape(1, -1, 2)
   

def LL2apx(lat, lon, zoom):
  """
  Converts lat/lon to pixel coordinates in given zoom of the EPSG:4326 pyramid
  [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244]
      Constant 20037508.342789244 comes from the circumference of the Earth in meters,
      which is 40 thousand kilometers, the coordinate origin is in the middle of extent.
      In fact you can calculate the constant as: 2 * math.pi * 6378137 / 2.0
      $ echo 180 85 | gdaltransform -s_srs EPSG:4326 -t_srs EPSG:900913
      Polar areas with abs(latitude) bigger then 85.05112878 are clipped off.

    What are zoom level constants (pixels/meter) for pyramid with EPSG:900913?

      whole region is on top of pyramid (zoom=0) covered by 256x256 pixels tile,
      every lower zoom level resolution is always divided by two
      initialResolution = 20037508.342789244 * 2 / 256 = 156543.03392804062
  180deg x 85.0511deg
  """
  originShift = 2 * math.pi * 6378137 / 2.0
  degx = lon * 1.0
  "log(tan($lat/2 + M_PI/4));"
  degy = math.log( math.tan((lat + 90 ) * math.pi / 360.0 )) *180/math.pi
  

  # my = my * originShift 
  
  
  tpx=256* (2**zoom)

  px = (degx+180)/360*tpx
  py = (-degy+180)/360*tpx
  print("lat:"+str(lat) + " lon:" +str(lon)+" tpx"+str(tpx)+"z:"+str(zoom)+" px:" + str(px) + " py:" + str(py))
  return px, py



def dist_filter_matches(kp1, kp2, matches, dist=100):
    mkp1, mkp2 = [], []
    for m in matches:
        distx=abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])
        disty=abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])  

        if distx <dist and  disty <dist :      
          mkp1.append( kp1[m.queryIdx] )
          mkp2.append( kp2[m.trainIdx] )      

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    print "distfilter", len(kp1), 'to',len(p1)
    return p1, p2, kp_pairs


 

def xformkp(k2a, H):
  """
  perspective transform cv2 keypoints
  """
  # l2 = np.array( [( (0,0) , (100,0) , (100,100) , (0,100) )], dtype=np.float32)   
  # mtx = cv2.getPerspectiveTransform(l2, l2)

  pkp = np.empty([len(k2a),2])
  nerror = 0
  for n in range(len(k2a)):
    # print 'xformkp k2a[n].pt' ,n, len(k2a), k2a[n].pt
    try:
      pkp[n] = [k2a[n].pt[0],k2a[n].pt[1]]
    except:
      nerror+=1 
      print 'xformkp problem',k2a[n] 
    if nerror>10:
      break
  # print 'before', pkp.shape  
  pkp = np.float32( cv2.perspectiveTransform(pkp.reshape(1, -1, 2), H).reshape(-1, 2)  )
  # print 'after', pkp.shape
  # quit()


  kp=[]
  for n in range(len(k2a)):

    k1 = k2a[n]
 
    # cv2.KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]]) 
    try:
      kp.append(cv2.KeyPoint(pkp[n][0], pkp[n][1], k1[2],  k1[3],  k1[4], int(k1[5]) ) )
    except:
      kp.append(cv2.KeyPoint(pkp[n][0], pkp[n][1],k2a[n].size ,k2a[n].angle,k2a[n].response,k2a[n].octave,k2a[n].class_id))

    try:
        minx=  min(pkp[n][0],minx)
        miny=  min(pkp[n][1],miny)        
        maxx=  max(pkp[n][0],maxx)
        maxy=  max(pkp[n][1],maxy)
    except:
        minx= pkp[n][0] 
        miny= pkp[n][1] 
        maxx= pkp[n][0] 
        maxy= pkp[n][1] 
  # print 'xformkp lens', len(k2a), len(kp)      
  return kp  , [minx,miny,maxx,maxy]


def xformp1(p1, H):
  """
  perspective transform  keypoints
  """
  # l2 = np.array( [( (0,0) , (100,0) , (100,100) , (0,100) )], dtype=np.float32)   
  # mtx = cv2.getPerspectiveTransform(l2, l2)
  p2 = np.zeros((len(p1),2))
  for n in range(len(p1)):
    p2[n] = [p1[n][0],p1[n][1]]

  pout = np.float32( cv2.perspectiveTransform(p2.reshape(1, -1, 2), H).reshape(-1, 2)   )
 
  return pout  


 

def translatekp(k2a, dx, dy):
  """
  perspective transform cv2 keypoints
  """
  # l2 = np.array( [( (0,0) , (100,0) , (100,100) , (0,100) )], dtype=np.float32)   
  # mtx = cv2.getPerspectiveTransform(l2, l2)
 
  for n in range(len(k2a)):
    k2a[n].pt[0] = k2a[n].pt[0] + dx
    k2a[n].pt[1] = k2a[n].pt[1] + dy
 
    try:
        minx=  min(k2a[n].pt[0],minx)
        miny=  min(k2a[n].pt[1],miny)        
        maxx=  max(k2a[n].pt[0],maxx)
        maxy=  max(k2a[n].pt[1],maxy)
    except:
        minx= k2a[n].pt[0]
        miny= k2a[n].pt[1]
        maxx= k2a[n].pt[0]
        maxy= k2a[n].pt[1] 
  print 'translatekp lens', len(k2a)
  return kp  , [minx,miny,maxx,maxy]


def translatep1(p1, dx, dy):
  """
  translate  points
  """
 
  H = np.eye( 3)
  H[0,2] = dx
  H[1,2] = dy

  return np.float32( cv2.perspectiveTransform(p1.reshape(1, -1, 2), H).reshape(-1, 2)   )
 





def Degrees2Radians(deg):
    """
    Converts deg to radians
    """
    return deg * math.pi / 180

def LL2Meters(lat, lon, floatpix=False):
    """
      Converts lat long  to meters using spherical projection
    """
    lat = Degrees2Radians(lat)
    lon = Degrees2Radians(lon)
    sinLat = math.sin(lat)
    x = earthRadius * lon
    y = earthRadius / 2 * math.log((1+sinLat)/(1-sinLat))
    return (x, y)

def MaxTiles(level):
    """
      Returns number of tiles at zoom level
    """
    return 1 << level

def MetersPerTile(level):
    """
      Returns meter width of each tile at equator
    """
    return earthCircumference / MaxTiles(level)

def MetersPerPixel(level):
    """
      Returns meter width of each pixel at equator
    """
    return MetersPerTile(level) / pixelsPerTile

# def Meters2Pixel(meters, level):
#     """
#     inputs
#       meters: in global projection
#       level: zoom level

#     output:
#       pixel pair 
#     """
#     metersPerPixel = MetersPerPixel(level)
#     x = (projectionOriginOffset + meters[0]) / metersPerPixel + 0.5
#     y = (projectionOriginOffset - meters[1]) / metersPerPixel + 0.5
#     return (x, y)

# def LL2Pixel(lat, lon, level, floatpix=False):
#     """
#     inputs
#       meters: in global projection
#       level: zoom level

#     output:
#       pixel pair 
#     """
#     return Meters2Pixel(LL2Meters(lat, lon), level)

def Pixel2Tile(pixel):
    x = int(pixel[0] / pixelsPerTile)
    y = int(pixel[1] / pixelsPerTile)
    return (x, y)

def LL2Tile(lat, lon, level):
    if lat < minLat:
        lat = minLat
    elif lat > maxLat:
        lat = maxLat
    return Pixel2Tile(LL2Pixel(lat, lon, level))

def Tile2Quadkey(tile, level):
    quadkey = ""
    for i in range(level, 0, -1):
        mask = 1 << (i-1)
        cell = 0
        if tile[0] & mask != 0:
            cell += 1
        if tile[1] & mask != 0:
            cell += 2
        quadkey += str(cell)
    return quadkey


def TXY2Quadkey(x,y, level):
    quadkey = ""
    for i in range(level, 0, -1):
        mask = 1 << (i-1)
        cell = 0
        if x & mask != 0:
            cell += 1
        if y & mask != 0:
            cell += 2
        quadkey += str(cell)
    return quadkey

def LL2Quadkey(lat, lon, level):
    return Tile2Quadkey(LL2Tile(lat, lon, level), level)

def Tile2LL(x,y,z):
 

    lon= (x/math.pow(2,z)*360-180) 
    n = PI-2*PI*y/math.pow(2,z)
    lat = (180/PI*math.atan(0.5*(math.exp(n)-math.exp(-n))))

    return (lat, lon)

def Pixel2LL(px,py,z):
 
    x=px/256
    y=py/256
    lon= (x/math.pow(2,z)*360-180) 
    n = PI-2*PI*y/math.pow(2,z)
    lat = (180/PI*math.atan(0.5*(math.exp(n)-math.exp(-n))))

    return (lat, lon)




def pixel2lonlat(px,py,z):
 
    x=px/256
    y=py/256
    lon= (x/math.pow(2,z)*360-180) 
    n = PI-2*PI*y/math.pow(2,z)
    lat = (180/PI*math.atan(0.5*(math.exp(n)-math.exp(-n))))

    return [lon, lat]



    
def Quadkey2Tile(quadkey):
    """Transform quadkey to tile coordinates"""
    tile_x, tile_y = (0,0)
    level = len(quadkey)
    for i in xrange(level):
        bit = level - i
        mask = 1 << (bit - 1)
        if quadkey[level - bit] == '1':
            tile_x |= mask
        if quadkey[level - bit] == '2':
            tile_y |= mask
        if quadkey[level - bit] == '3':
            tile_x |= mask
            tile_y |= mask
    return [(tile_x, tile_y), level]


 

def reproject(longitude, latitude ):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) 
                for lat, long in zip(latitude, longitude)]
    return x, y

def lonlat2m(coords ):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    out =[]
    for lonlat in coords:
        y = lonlat[1]*lat_dist
        x = lonlat[0] * lat_dist * cos(radians(lonlat[1])) 
        out.append([x,y])
    return out


def area_of_polygon(x, y):
    """Calculates the area of an arbitrary polygon given its vertices"""
    area = 0.0
    for i in xrange(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return (-area) / 2.0


def arealonlat(coords):
    area = 0.0
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    xy =[]
    for lonlat in coords:
        y = lonlat[1]*lat_dist
        x = lonlat[0] * lat_dist * cos(radians(lonlat[1])) 
        xy.append([x,y])


    for i in xrange(-1, len(xy)-1):
        area += xy[i][0] * (xy[i+1][1] - xy[i-1][1])
    return (-area) / 2.0

    return area


def calcbearing(pointA, pointB):
    """
    Calculates the bearing between two points.
 
    The formulae used is the following
 
    :Parameters:
      -  pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      -  pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
 
    :Returns:
      The bearing in degrees
 
    :Returns Type:
      float
    """
    # if (type(pointA) != tuple) or (type(pointB) != tuple):
    #     raise TypeError("Only tuples are supported as arguments")
 
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
 
    diffLong = math.radians(pointB[1] - pointA[1])
 
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))
 
    initial_bearing = math.atan2(x, y)
 
    # Now we have the initial bearing but math.atan2 return values
    # from which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
 
    return compass_bearing

def calcdistlatlon(pointA, pointB):
    R = 6373.0 #km

    lat1 = math.radians(pointA[0])
    lon1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[0])
    lon2 = math.radians(pointB[1])


    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c #km

    return distance
 

def calcdistlonlat(pointA, pointB):
    R = 6373.0 #km
    lat1 = math.radians(pointA[1])
    lon1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[1])
    lon2 = math.radians(pointB[0])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c #km

    return distance

def calcxydist(pointA, pointB):
 
    dx = pointA[0] - pointB[0]
    dy = pointA[1] - pointB[1]

    c =math.sqrt(dx*dx + dy*dy)
 

    return c


def anglecos (a,b,c): return math.acos((c**2-b**2-a**2)/(-2*a*b))

def anglecosdeg (a,b,c): return math.acos((c**2-b**2-a**2)/(-2*a*b))*180/3.14159

 




def reproject(longitude, latitude ):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) 
                for lat, long in zip(latitude, longitude)]
    return x, y

def areap(x, y):
    """Calculates the area of an arbitrary polygon given its vertices"""
    area = 0.0
    for i in xrange(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return (-area) / 2.0


def area2(xy):
    """Calculates the area of an arbitrary polygon given its vertices"""
    area = 0.0
    for i in xrange(-1, len(xy)-1):
        area += xy[i][0] * (xy[i+1][1] - xy[i-1][1])
    return (area) / 2.0


def LL2area(longitude, latitude ):
    x,y = reproject(longitude, latitude )
    return areap(x, y)










 






def qk2xy(quadkey):
    """Transform quadkey to tile coordinates"""
    tile_x, tile_y = (0,0)
    level = len(quadkey)
    for i in xrange(level):
        bit = level - i
        mask = 1 << (bit - 1)
        if quadkey[level - bit] == '1':
            tile_x |= mask
        if quadkey[level - bit] == '2':
            tile_y |= mask
        if quadkey[level - bit] == '3':
            tile_x |= mask
            tile_y |= mask
    return  tile_x, tile_y, level


def qk2pxpy(quadkey):
    """Transform quadkey to tile coordinates"""
    tile_x, tile_y = (0,0)
    level = len(quadkey)
    for i in xrange(level):
        bit = level - i
        mask = 1 << (bit - 1)
        if quadkey[level - bit] == '1':
            tile_x |= mask
        if quadkey[level - bit] == '2':
            tile_y |= mask
        if quadkey[level - bit] == '3':
            tile_x |= mask
            tile_y |= mask
    return  tile_x*256.0, tile_y*256.0, level



def t2lonlat(x,y,z):
 

    lon= (x/math.pow(2,z)*360-180) 
    n = PI-2*PI*y/math.pow(2,z)
    lat = (180/PI*math.atan(0.5*(math.exp(n)-math.exp(-n))))

    return (lon, lat)


def xy2lonlat(x,y,z):
   

    lon= (x/math.pow(2,z)*360-180) 
    n = PI-2*PI*y/math.pow(2,z)
    lat = (180/PI*math.atan(0.5*(math.exp(n)-math.exp(-n))))

    return lon, lat


def tileres(z,lat=38.0):
    qka=lonlat2qk(0.0,lat,z)
    qkb=qk2qk(qka,1,0)
    la=qk2lonlat(qka)
    lb=qk2lonlat(qkb)
    dist=calcdistlonlat(la,lb)
    return dist*1000.0/256


def qk2lonlat(quadkey):
    x, y, z=qk2xy(quadkey)
    return t2lonlat(x,y,z)


# def lonlat2pxpy(lon, lat, zoom, floatpix=False):
#     px,py = LL2Pixel( lat,lon, zoom)
#     return px,py

def lonlat2xy(lon, lat, zoom):
    px,py = lonlat2pxpy( lon,lat, zoom)
    return int(math.floor(px/256)), int(math.floor(py/256))

def lonlat2qk(lon, lat, zoom):
    x,y = lonlat2xy(lon,lat,  zoom)
    qk = xy2qk(x,y, zoom)
    return qk


# def latlon2qk( lat, lon,zoom):
#     qk = LL2Quadkey(lat, lon, zoom)
#     return qk

def xy2qk(x,y,z):
    return Tile2Quadkey((x, y), z)

 

def svginit(lon,lat, zoom):
    Config.tlpx, Config.tlpy = lonlat2pxpy(lon,lat, zoom)
    Config.svgzoom=zoom
    Config.svg=''

def qk2qk( qk, dx, dy):
    x, y, z = qk2xy(qk)
    return xy2qk(x+dx, y+dy, z)





def svgqk(qk, label="", stroke='rgba(0,0,0,0.5)', fill='rgba(0,255,255,0.1)', width='0.1', opacity=1.0, textsize = "6", textcolor ="rgba(0,0,0,0.6)"):
    x, y, z = qk2xy(qk)
    lon1, lat1 = t2lonlat(x, y, z)
    lon2, lat2 = t2lonlat(x+1, y+1, z)
    svgrect(lon1,lat1,lon2,lat2, stroke, fill, width, opacity)
    svgtxtll( label, lon1, lat1, size=textsize, color=textcolor)


def svgpqk(qk, label="q", stroke='rgba(0,0,0,0.5)', fill='rgba(0,255,255,0.1)', width='0.1', opacity=1.0, textsize = "6", textcolor ="rgba(0,0,0,0.6)"):
    x, y, z = qk2xy(qk)
    lona, lata = t2lonlat(x, y, z)
    lonb, latb = t2lonlat(x+1, y+1, z)
    pxa, pya = lonlat2pxpy(lona,lata, Config.svgzoom)
    pxb, pyb = lonlat2pxpy(lonb,latb, Config.svgzoom)
    svgprect(pxa,pya,pxb,pyb, stroke, fill, width, opacity)
    # svgtxtpll( label, lon1, lat1, size=textsize, color=textcolor)

def svgpqk(qk, displacements ={"tldx" : 0, "tldy" : 0, "trdx" : 0, "trdy" : 0, "bldx" : 0, "bldy" : 0, "brdx" : 0, "brdy" : 0}
, label="q", stroke='rgba(255,0,0,1)', fill='rgba(0,255,255,0.1)', width='0.1', opacity=1.0, textsize = "6", textcolor ="rgba(0,0,0,0.6)"):
    x, y, z = qk2xy(qk)
    lona, lata = t2lonlat(x, y, z)
    lonb, latb = t2lonlat(x+1, y+1, z)
    pxa, pya = lonlat2pxpy(lona,lata, Config.svgzoom)
    pxb, pyb = lonlat2pxpy(lonb,lata, Config.svgzoom)
    pxc, pyc = lonlat2pxpy(lonb,latb, Config.svgzoom)
    pxd, pyd = lonlat2pxpy(lona,latb, Config.svgzoom)

    # svgprect(pxa,pya,pxb,pyb, stroke, fill, width, opacity)
    d=displacements
    scale=math.pow(2, 18-Config.svgzoom)
    # llarr=[[-122.446, 37.80],[-122.445, 37.79], [-122.4, 37.76]]
    pxarr=[[pxa+d['tldx']/scale, pya+d['tldy']/scale], [pxb+d['trdx']/scale, pyb+d['trdy']/scale], [pxc+d['brdx']/scale, pyc+d['brdy']/scale], [pxd+d['bldx']/scale, pyd+d['bldy']/scale] ]


    npts = len(pxarr)
    out ="" 
    tlpx=Config.tlpx
    tlpy=Config.tlpy
    for n in xrange(0,npts):
        npta=pxarr[n]
        px,py =  npta[0],npta[1] 
        x1 = str(px-tlpx)
        y1 = str(py-tlpy)
        print tlpx, tlpy, px, py


        out+=' '+x1+','+y1+''
    Config.svg+='<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';"  />'



    # svgpoly(llarr, stroke='rgba(0,0,0,0.5)', fill='rgba(0,0,255,0.1)', width='100')
    # svgtxtpll( label, lon1, lat1, size=textsize, color=textcolor)


def svgqkcen(qk, lonc,latc, label="", stroke='rgba(0,0,0,0.2)', fill='rgba(0,255,255,0.1)', width='0.1'):
    x, y, z =  qk2xy(qk)
    lon1, lat1 = t2lonlat(x, y, z)
    lon2, lat2 = t2lonlat(x+1, y+1, z)
    svgrect(lon1,lat1,lon2,lat2, stroke, fill, width)
    llarr=[[lonc,latc],[lon1,lat1] ]
    svgll(llarr, color=stroke, width=width, close=False)
    frac=0.2
    svgtxtll(qk, lon1,  lat2*frac + lat1*(1-frac) , size="5", color='rgba(255,180,80,1)')
    frac=0.4
    svgtxtll(label, lon1,  lat2*frac + lat1*(1-frac) , size="5", color='rgba(255,180,60,1)')


def svgqkimg(qk, stroke='rgba(0,0,0,0.5)', fill='rgba(0,255,255,0.1)', width='0.1', opacity=0.5):
    x, y, z = qk2xy(qk)
    lon1, lat1 = t2lonlat(x, y, z)
    lon2, lat2 = t2lonlat(x+1, y+1, z)
    tlx=Config.tlpx
    tly=Config.tlpy
    pxa,pya = lonlat2pxpy( lon1,lat1, Config.svgzoom)
    pxb,pyb = lonlat2pxpy( lon2,lat2, Config.svgzoom)
 

    x0 = str(pxa-tlx)
    y0 = str(pya-tly)
    w = str(pxb-pxa)
    h = str(pyb-pya)
    img='http://t0.tiles.virtualearth.net/tiles/a'+qk+'.jpeg?g=1398'
    Config.svg+='<image xlink:href="'+img+'" x="'+x0+'px" y="'+y0+'px" height="'+h+'px" width="'+w+'px" style="z-index:-10; opacity:'+str(opacity)+'"/>'

    frac=0.2
    svgtxtll(qk, lon1,  lat2*frac + lat1*(1-frac) , size="6", color='rgba(0,180,60,1)')

    # svgrect(lon1,lat1,lon2,lat2, stroke, fill, width)
# 
def svgqkai(qk, diri="", stroke='rgba(0,0,0,0.5)', fill='rgba(0,255,255,0.1)', width='0.1',  opacity=1):
    x, y, z = qk2xy(qk)
    lon1, lat1 = t2lonlat(x, y, z)
    lon2, lat2 = t2lonlat(x+1, y+1, z)
    tlx=Config.tlpx
    tly=Config.tlpy
    pxa,pya = lonlat2pxpy( lon1,lat1, Config.svgzoom)
    pxb,pyb = lonlat2pxpy( lon2,lat2, Config.svgzoom)
 
    x0 = str(pxa-tlx)
    y0 = str(pya-tly)
    w = str(pxb-pxa)
    h = str(pyb-pya)
 
    # svgrect(lon1,lat1,lon2,lat2, stroke, fill, width)
    img = diri + '/a' + qk + '.jpg'
    Config.svg += '<image xlink:href="'+img+'" x="'+x0+'px" y="'+y0+'px" height="'+h+'px" width="'+w+'px" style="z-index:-10; opacity:'+str(opacity)+';"/>'+'\n'

def svgtxtll(txt, lon, lat, size="20", color="#000"):
    tlx=Config.tlpx
    tly=Config.tlpy
    pxa,pya = lonlat2pxpy( lon,lat, Config.svgzoom)

    x1 = str(pxa-tlx)
    y1 = str(pya-tly)
    Config.svg+='<text x="'+x1+'" y="'+y1+'" style="font-family: helvetica, sans-serif; font-weight: normal; font-style: normal" font-size="'+size+'px" fill="'+color+'">'+txt+'</text>'


def svgpoly(llarr, stroke='rgba(0,0,0,0.5)', fill='rgba(255,0,255,0.1)', width='1', latlon=False):
    # <polygon points="60,20 100,40 100,80 60,100 20,80 20,40"/>
    tlx=Config.tlpx
    tly=Config.tlpy

    npts=len(llarr)
    out=""
    for n in xrange(0,npts):
        npta=llarr[n]
        # print "npta", npta[0],npta[1]
        pxa,pya = lonlat2pxpy( npta[0],npta[1], Config.svgzoom)
        if latlon: pxa,pya = lonlat2pxpy( npta[1],npta[0], Config.svgzoom)
        x1 = str(pxa-tlx)
        y1 = str(pya-tly)
        out+=' '+x1+','+y1+''
 
    # Config.svg+='<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';"  />'
    return '<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';"  />'


 


def svgrect(lon1,lat1,lon2,lat2, stroke='rgba(0,0,0,0.5)', fill='rgba(0,0,255,0.1)', width='1', opacity=1.0):
    # <polygon points="60,20 100,40 100,80 60,100 20,80 20,40"/>
    tlx=Config.tlpx
    tly=Config.tlpy
    pxa,pya = lonlat2pxpy( lon1,lat1, Config.svgzoom)
    pxb,pyb = lonlat2pxpy( lon2,lat2, Config.svgzoom)
    out=""

    x0 = str(pxa-tlx)
    y0 = str(pya-tly)
    x1 = str(pxb-tlx)
    y1 = str(pya-tly)
    x2 = str(pxb-tlx)
    y2 = str(pyb-tly)
    x3 = str(pxa-tlx)
    y3 = str(pyb-tly)
    out+=' '+x0+','+y0+' '+x1+','+y1+' '+x2+','+y2+' '+x3+','+y3+''
    fout = '<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';opacity:'+str(opacity)+';"  />'
    Config.svg+=fout
    return fout

def svgrectstr(lon1,lat1,lon2,lat2, stroke='rgba(0,0,0,0.5)', fill='rgba(0,0,255,0.1)', width='1', opacity=1.0,  lon0=0, lat0= 0):
    # <polygon points="60,20 100,40 100,80 60,100 20,80 20,40"/>
 
    tlx, tly = lonlat2pxpy( lon0,lat0, Config.svgzoom)    
  
    pxa,pya = lonlat2pxpy( lon1,lat1, Config.svgzoom)
    pxb,pyb = lonlat2pxpy( lon2,lat2, Config.svgzoom)
    out=""

    x0 = str(pxa-tlx)
    y0 = str(pya-tly)
    x1 = str(pxb-tlx)
    y1 = str(pya-tly)
    x2 = str(pxb-tlx)
    y2 = str(pyb-tly)
    x3 = str(pxa-tlx)
    y3 = str(pyb-tly)
    out+=' '+x0+','+y0+' '+x1+','+y1+' '+x2+','+y2+' '+x3+','+y3+''
    fout = '<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';opacity:'+str(opacity)+';"  />'
 
    return fout


def svgprect(pxa,pya,pxb,pyb, stroke='rgba(0,0,0,0.5)', fill='rgba(0,0,255,0.1)', width='1', opacity=1.0):
    # <polygon points="60,20 100,40 100,80 60,100 20,80 20,40"/>
    tlpx=Config.tlpx
    tlpy=Config.tlpy 
    x0 = str(pxa-tlpx)
    y0 = str(pya-tlpy)
    x1 = str(pxb-tlpx)
    y1 = str(pya-tlpy)
    x2 = str(pxb-tlpx)
    y2 = str(pyb-tlpy)
    x3 = str(pxa-tlpx)
    y3 = str(pyb-tlpy)
    out=' '+x0+','+y0+' '+x1+','+y1+' '+x2+','+y2+' '+x3+','+y3+''
 
    Config.svg+='<polygon points="'+out+'"  style="fill:'+fill+'; stroke:'+stroke+';stroke-width:'+width+';opacity:'+str(opacity)+';"  />'




def svgll(llarr, color='rgba(0,0,255,0.8)', width='1', close=False, latlon=False):

    tlx=Config.tlpx
    tly=Config.tlpy

    npts=len(llarr)
    out=""
    c=1
    if close: c=0
    for n in xrange(0,npts-c):
        npta=llarr[n]
        nptb=llarr[(n+1)%npts]

        if latlon==True:
            pxa,pya = lonlat2pxpy( npta[1],npta[0], Config.svgzoom)
            pxb,pyb = lonlat2pxpy( nptb[1],nptb[0], Config.svgzoom)
        else:
            pxa,pya = lonlat2pxpy( npta[0],npta[1], Config.svgzoom)
            pxb,pyb = lonlat2pxpy( nptb[0],nptb[1], Config.svgzoom)    
        # print "pxxa",pxa-tlx,pya-tly, pxa-tlx,pya-tly
        # "drw.line( (pxa-tlx,pya-tly, pxb-tlx,pyb-tly ), color , width=1)
        x1 = str(pxa-tlx)
        y1 = str(pya-tly)
        x2 = str(pxb-tlx)
        y2 = str(pyb-tly)

        out+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+color+'" stroke-width="'+width+'" />' 
    Config.svg+=out
    return 



def svgpll(pxarr, color='rgba(0,0,255,0.1)', width='1', close=False):

    tlpx=Config.tlpx
    tlpy=Config.tlpy
    npts=len(pxarr)
    out=""
    c=1
    if close: c=0
    for n in xrange(0,npts-c):
        npta=pxarr[n]
        nptb=pxarr[(n+1)%npts]
        pxa,pya = lonlat2pxpy( npta[0],npta[1], Config.svgzoom)
        pxb,pyb = lonlat2pxpy( nptb[0],nptb[1], Config.svgzoom)
        # print "pxxa",pxa-tlx,pya-tly, pxa-tlx,pya-tly
        # "drw.line( (pxa-tlx,pya-tly, pxb-tlx,pyb-tly ), color , width=1)
        x1 = str(pxa-tlx)
        y1 = str(pya-tly)
        x2 = str(pxb-tlx)
        y2 = str(pyb-tly)

        out+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+color+'" stroke-width="'+width+'" />'
    Config.svg+=out
    return 




def writesvg(fn, w=1600, h=1200):
    svg0='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="'+str(w)+'" height="'+str(h)+'">'
  
    out = svg0+Config.svg+'</svg>'

    with open(fn, "w") as myfile:
      myfile.write(out)
    print "wrote", fn  

def svgwrite(fn, w=1600, h=1200):
    svg0='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" lala="" width="'+str(w)+'" height="'+str(h)+'">'
  
    out = svg0+Config.svg+'</svg>'

    with open(fn, "w") as myfile:
      myfile.write(out)



def pnames(path):
    fsplit=path.strip().split('/')
    lenf = len(fsplit)
    sname = fsplit[lenf-4]+"_"+fsplit[lenf-3]+"_"+fsplit[lenf-2]+"_"+fsplit[lenf-1]
    data = sname.replace(".JPG","")
    seqnum = 1000 * int(fsplit[-2][1:3]) + int(path[-8:-4])
    base = path[:-21]
 

    out = {"path":path, 'base': base, 'data':data, "iname": fsplit[-1] , "inum":  seqnum,'d3200':fsplit[-2], 'date' : fsplit[-4], "im15":data+"/s"+ fsplit[-1] }
    return out


def genpath(imnum, base):
    d = imnum%1000
    d32 = (imnum - d)/1000
    out =  base + '1'+'%02d'%d32 + 'D3200/DSC_'+'%04d'%d+'.JPG'
    if d == 0 and d32 == 0: out =  base + '100D3200/DSC_0001.JPG'
    if d == 0 and d32 > 0: 
        out =  base + '1'+'%02d'%d32 + 'D3200/DSC_0001.JPG'
    return out



def point_inside_polygon(x,y,poly):

  return pointinpoly(x,y,poly)
 
def pointinpoly(x,y,poly):
    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
  
    return inside



def corninpoly(poly=[], corn=[]):
  out = []
  for c in corn:
    if pointinpoly(c[0],c[1],poly):
      out.append(c)

  return out    



def plotvepic(qklist=[], svgfn='kpvepic.svg'):
  svgout=''


  qklist0=[]
  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  
  tx0, ty0, z=qk2xy(qklist[0] )  
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx
      dya=ty
      # print qk
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str( (dxa-tx0)*256)+'px" y="'+str((dya-ty0)*256)+'px" height="256px" width="256px" style=" opacity:1;"   dxa="'+str(dxa)+'" />';
 

  svgo='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform=" translate(0,0)">' + svgout + '</g></svg>'
  with open(svgfn, mode='w') as ff:
        ff.write(svgo )    




    
def reproject(longitude, latitude ):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) 
                for lat, long in zip(latitude, longitude)]
    return x, y

def areap(x, y):
    """Calculates the area of an arbitrary polygon given its vertices"""
    area = 0.0
    for i in xrange(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return (-area) / 2.0



def opensvg(fn, w=1600, h=1200):
    svg0='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="'+str(w)+'" height="'+str(h)+'">'

    with open(fn, "w") as myfile:
      myfile.write(svg0)
 

def appendsvg(fn, data):
    with open(fn, "a") as myfile:
      myfile.write(data)
 
def appendfile(fn, data):
    with open(fn, "a") as myfile:
      myfile.write(data)


def writefile(fn, data):
    with open(fn, "w") as myfile:
      myfile.write(data)


def closesvg(fn, w=1600, h=1200):
 

    with open(fn, "a") as myfile:
      myfile.write( '</svg>')
    print "wrote", fn  



def clahergb(img,cr=0.7,cg=0.7,cb=0.7):
    img=img.astype(np.uint8)
    grid = (8,8)
    r, g, b = cv2.split(img)
    claher = cv2.createCLAHE(cr, tileGridSize=grid)
    claheg = cv2.createCLAHE(cg, tileGridSize=grid)
    claheb = cv2.createCLAHE(cb, tileGridSize=grid)
     
    cl1 = claher.apply(r)
    a1 = claheg.apply(g)
    b1 = claheb.apply(b)
    img2 = cv2.merge((cl1, a1, b1 ))
    return img2





def cornlonlat2pixels(corners, zoom=18):
  rc=[]
  for c in corners:
    pxb, pyb = lonlat2pxpy(c[0],c[1], zoom)
    rc.append([pxb, pyb] )
  # return  np.float32(rc)
  return   rc




def getstatus(imuri, stage='georefdone'):
  r =  requests.get("http://10.0.0.4/repo/getstatus.php?stage="+stage+"&imuri="+imuri )     

  print r.url
  try:  
    jsons=json.loads(r.text)
    return jsons
  except:
    return r.text

 
def jloads(a):
  return  json.loads(a)
def jdumps(a):
  return  json.dumps(a)


def cornlimits(corners ):

  xmin = min(corners, key = lambda t: t[0])[0]
  xmax = max(corners, key = lambda t: t[0])[0]
  ymin = min(corners, key = lambda t: t[1])[1]
  ymax = max(corners, key = lambda t: t[1])[1]
 

  return [xmin, xmax, ymin, ymax]





def cornalimits(cornersa=[]):

  for corners in cornersa:
    try:
      xmin = min(xmin, min(corners, key = lambda t: t[0])[0])
      xmax = max(xmax, max(corners, key = lambda t: t[0])[0])
      ymin = min(ymin, min(corners, key = lambda t: t[1])[1])
      ymax = max(ymax, max(corners, key = lambda t: t[1])[1])
    except:
      xmin = min(corners, key = lambda t: t[0])[0]
      xmax = max(corners, key = lambda t: t[0])[0]
      ymin = min(corners, key = lambda t: t[1])[1]
      ymax = max(corners, key = lambda t: t[1])[1]


  try:
    aa =     [xmin, xmax, ymin, ymax] 
  except:
    print 'cornalimits corners no good', cornersa
    return []

  return aa    



def cornabox(cornersa=[]):
  for corners in cornersa:
    try:
      xmin = min(xmin, min(corners, key = lambda t: t[0])[0])
      xmax = max(xmax, max(corners, key = lambda t: t[0])[0])
      ymin = min(ymin, min(corners, key = lambda t: t[1])[1])
      ymax = max(ymax, max(corners, key = lambda t: t[1])[1])
    except:
      xmin = min(corners, key = lambda t: t[0])[0]
      xmax = max(corners, key = lambda t: t[0])[0]
      ymin = min(corners, key = lambda t: t[1])[1]
      ymax = max(corners, key = lambda t: t[1])[1]

  return [[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin] ]    


 

def corn2qklist(corners, zoom):
    clim = cornlimits(corners )
    # t00=LL2Tile(max(cornll[0,:,1]), min(cornll[0,:,0])  , zoom)  
    # t11=LL2Tile(min(cornll[0,:,1]) , max(cornll[0,:,0])  , zoom)  
    tx0, ty0= lonlat2xy(clim[0],clim[3]  , zoom)  
    tx1, ty1= lonlat2xy(clim[1],clim[2]  , zoom)  
 
 
    height=ty1-ty0+1
    width=tx1-tx0+1

    # if height >30 or width > 30:
    #   
    #   return []

    print 'conrqklist', width, height
    qklist = []
    ntiles=0
    for dy in xrange(0,height):
      for dx in xrange(0,width):
        qkreq =  Tile2Quadkey((tx0+dx, ty0+dy), zoom)
        qklist.append(qkreq)
    return qklist

def corna2qklist(corners, zoom):
    # print 'corna2qklist corner sets' , len(corners), len(corners[0]) 

 
    clim = cornalimits(corners )
    # print 'corna2qklist clim',clim
    # t00=LL2Tile(max(cornll[0,:,1]), min(cornll[0,:,0])  , zoom)  
    # t11=LL2Tile(min(cornll[0,:,1]) , max(cornll[0,:,0])  , zoom)  
    tx0, ty0= lonlat2xy(clim[0],clim[3]  , zoom)  
    tx1, ty1= lonlat2xy(clim[1],clim[2]  , zoom)  
 
 
    height=ty1-ty0+1
    width=tx1-tx0+1

    # if height >50 or width > 50:
    #   print 'conrqklist too big', width, height
    #   return []


    qklist = []
    ntiles=0
    for dy in xrange(0,height):
      for dx in xrange(0,width):
        qkreq =  Tile2Quadkey((tx0+dx, ty0+dy), zoom)
        qklist.append(qkreq)
    return qklist


 

def cornqklist(corners, zoom):
    return corn2qklist(corners, zoom)
     





def plotkpold(kparray, svgfn='kp.svg', x0=0, y0=0, poly = []):
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">') 

  style=['stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(0,200,0,0.8); fill:rgba(0,255,155,0.2);stroke-width:1;',
    'stroke:rgba(0,0,220,0.8); fill:rgba(55,55,220,0.2);stroke-width:1;',
    'stroke:rgba(255,0,0,0.8); fill:rgba(255,0,250,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(155,0,250,0.2);stroke-width:1;',
    'stroke:rgba(25,110,110,0.8); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]

  ns=0  
  for k2a in kparray: 
    st=  style[ns%len(style)]
    ns+=1
    for n in range(len(k2a)):
      x1 , y1 = k2a[n].pt[0], k2a[n].pt[1]
 
  #   # cv2.line(vis, (x1, y1), (x2, y2), green)
      with open(svgfn, mode='a') as ff:
        ff.write('<circle cx="'+str(x1+x0)+'" cy="'+str(y1+y0)+'" r="'+str(ns+1)+'" style="'+st+'"  />' )
  
  for pol1 in poly:
    p=''
    # print pol1
    for c in pol1:
      p += " "+str(c[0]+x0)+","+str(c[1]+y0)+""
  
    with open(svgfn, mode='a') as ff:

      ff.write('<polygon points="'+p+'"  style="fill:rgba(255,255,0,0.2); stroke:rgba(5,0,30,0.8);stroke-width:1px;"  />')
  

  # svgwrite( '<g transform="translate(10,10)"><image xlink:href="'+Config.imaurl+'" x="0px" y="0px"  height="'+str(Config.imah)+'px" width="'+str(Config.imaw)+'px"  style=" opacity:1;"/></g>'+'\n' )
  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  print 'outsvg', svgfn    






def plotmatches(mo, svgfn='kpmatch.svg' ):
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">') 

  style=['stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(0,200,0,0.8); fill:rgba(0,255,155,0.2);stroke-width:1;',
    'stroke:rgba(0,0,220,0.8); fill:rgba(55,55,220,0.2);stroke-width:1;',
    'stroke:rgba(255,0,0,0.8); fill:rgba(255,0,250,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(155,0,250,0.2);stroke-width:1;',
    'stroke:rgba(25,110,110,0.8); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]

  ns=0  

  for mm in mo['matches']: 

    kpa = mo['kpa'][mm.queryIdx]
    kpb = mo['kpb'][mm.trainIdx]
    try:
        xa , ya  = kpa.pt[0]-x0, kpa.pt[1]-y0
        xb , yb  = kpb.pt[0]-x0, kpb.pt[1]-y0
    except:
        x0=kpa.pt[0]-1000
        y0=kpa.pt[1]-1000
        xa , ya  = kpa.pt[0]-x0, kpa.pt[1]-y0
        xb , yb  = kpb.pt[0]-x0, kpb.pt[1]-y0
    aa='<circle cx="'+str(xa )+'" cy="'+str(ya)+'" r="'+str(2)+'" style="stroke:rgba(220,0,0,0.4); fill:rgba(255,0,0,0.2);stroke-width:1;"  />' 
    
    aa+='<circle cx="'+str(xb )+'" cy="'+str(yb)+'" r="'+str(2)+'" style="stroke:rgba(0,0,220,0.4); fill:rgba(0,255,0,0.2);stroke-width:1;"  />'     
    aa+='<line x1="'+str(xa)+'" y1="'+str(ya)+'" x2="'+str(xb)+'" y2="'+str(yb)+'" style="stroke:rgba(0,0,60,0.1); stroke-width:1;"  />' 
    with open(svgfn, mode='a') as ff:
        ff.write(aa)
 
  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  print 'outsvg', svgfn    




def plotpairs(mo, svgfn='kpmatch.svg', dx=1000,dy=1000):
  '''
  {p1, p2, corners}
  '''
  with open(svgfn, mode='w') as file:
    file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">') 

  style=['stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
  'stroke:rgba(0,200,0,0.8); fill:rgba(0,255,155,0.2);stroke-width:1;',
  'stroke:rgba(0,0,220,0.8); fill:rgba(55,55,220,0.2);stroke-width:1;',
  'stroke:rgba(255,0,0,0.8); fill:rgba(255,0,250,0.2);stroke-width:1;',
  'stroke:rgba(0,0,0,0.8); fill:rgba(155,0,250,0.2);stroke-width:1;',
  'stroke:rgba(25,110,110,0.8); fill:rgba(255,0,150,0.2);stroke-width:1;',
  'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
  ]
  x0=0
  y0=0


  for np in range(len(mo['p1'])): 

      kpa = mo['p1'][np]
      kpb = mo['p2'][np]
      if kpa[0]>5000:
          x0=kpa[0]
          y0=kpa[1]
      xa , ya  = kpa[0]-x0+dx, kpa[1]-y0+dy
      xb , yb  = kpb[0]-x0+dx, kpb[1]-y0+dy


      aa='<circle cx="'+str(xa )+'" cy="'+str(ya)+'" r="'+str(2)+'" style="stroke:rgba(220,0,0,0.4); fill:rgba(255,0,0,0.2);stroke-width:1;"  />' 

      aa+='<circle cx="'+str(xb )+'" cy="'+str(yb)+'" r="'+str(2)+'" style="stroke:rgba(0,0,220,0.4); fill:rgba(0,255,0,0.2);stroke-width:1;"  />'     
      aa+='<line x1="'+str(xa)+'" y1="'+str(ya)+'" x2="'+str(xb)+'" y2="'+str(yb)+'" style="stroke:rgba(0,0,60,0.1); stroke-width:1;"  />' 
      with open(svgfn, mode='a') as ff:
          ff.write(aa)
  
  try:
      clist = mo['corners']
  except:
      clist =[]

  for cn in  range(len(clist)):
      corns = clist[cn]
      print 'clist',corns
      p=''
      for c in corns:
          p += " "+str(c[0]+x0+dx)+","+str(c[1]+y0+dy)+""

      with open(svgfn, mode='a') as ff:
        ff.write('<polygon points="'+p+'"  style="'+style[cn%len(style)]+'"  />')

              
  with open(svgfn, mode='a') as file:
    file.write('</g></svg>') 
  print 'out plotpairs', svgfn    


def plotkp(kparray, svgfn='kp.svg', x0=0, y0=0, poly = [], tiles=[], scale=1.0, rad=2):
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale('+str(scale)+')">') 
  if len(tiles)>0:
    aa='' 
    tx0, ty0, z = qk2xy(tiles[0])
    for tile in tiles:
        tx, ty, z = qk2xy(tile)
        aa+='<g transform="translate('+str((tx-tx0)*256)+','+str((ty-ty0)*256)+')"><image xlink:href="http://t0.tiles.virtualearth.net/tiles/a'+ tile+'.jpeg?g=1398" x="0px" y="0px"  height="256px" width="256px"  style=" opacity:1;"/> </g>'+'\n'
    with open(svgfn, mode='a') as ff:
      ff.write(aa)
  style=['stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(0,200,0,0.8); fill:rgba(0,255,155,0.2);stroke-width:1;',
    'stroke:rgba(0,0,220,0.8); fill:rgba(55,55,220,0.2);stroke-width:1;',
    'stroke:rgba(255,0,0,0.8); fill:rgba(255,0,250,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(155,0,250,0.2);stroke-width:1;',
    'stroke:rgba(25,110,110,0.8); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]

  ns=0  
  for k2a in kparray: 
    st=  style[ns%len(style)]
    ns+=1
    for n in range(len(k2a)):
      x1 , y1 = k2a[n].pt[0], k2a[n].pt[1]
 
  #   # cv2.line(vis, (x1, y1), (x2, y2), green)
      with open(svgfn, mode='a') as ff:
        ff.write('<circle cx="'+str(x1+x0)+'" cy="'+str(y1+y0)+'" r="'+str(ns+rad)+'" style="'+st+'"  />' )
  
  for pol1 in poly:
    p=''
    # print pol1
    for c in pol1:
      p += " "+str(c[0]+x0)+","+str(c[1]+y0)+""
  
    with open(svgfn, mode='a') as ff:
      ff.write('<polygon points="'+p+'"  style="fill:rgba(255,255,0,0.2); stroke:rgba(5,0,30,0.8);stroke-width:1px;"  />')




  

  # svgwrite( '<g transform="translate(10,10)"><image xlink:href="'+Config.imaurl+'" x="0px" y="0px"  height="'+str(Config.imah)+'px" width="'+str(Config.imaw)+'px"  style=" opacity:1;"/></g>'+'\n' )
  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  print 'outsvg', svgfn    



def mosaicfeaturescenters(centers, z=16):
 
  mins = np.amin(centers, axis=0)
  ptps = np.ptp(centers,axis=0)

  minlon = mins[0]
  maxlat = mins[1]+ptps[1]

  print "mosaicfeaturescenters", minlon,maxlat
  tiles=[]
  txmin1, tymin1 =lonlat2xy(mins[0],mins[1]+ptps[1], z)
  txmax1, tymax1 =lonlat2xy(mins[0]+ptps[0],mins[1], z)


  print 'w h', txmax1-txmin1, tymax1-tymin1

  qks=lonlat2qk(mins[0],mins[1]+ptps[1],z)
  qk0 = qk2qk(qks,-2,-2)
  wqk = txmax1-txmin1+5
  hqk = tymax1-tymin1+5
  txmin, tymin, z =qk2xy(qk0)
  qklist=[]

  for yy in range(hqk):
    for xx in range(wqk):
      qklist.append(qk2qk(qk0,xx,yy))

 
  for qk in qklist:
    txq,tyq, z = qk2xy(qk)   
    tx = txq - txmin 
    ty = tyq - tymin 
    binb =  Config.vefeat + 'a'+qk + Config.ext + ".bin"
    k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)

    try:
      # fail on first iter
      ddescb = np.append(ddescb,d, axis=0)
      k2b.extend(k)
    except:
      # first iter
      binb =  Config.vefeat + 'a'+qk + Config.ext + ".bin"
      k2b, ddescb = getfeat(binb, dx = 0, dy = 0)
          

    tiles.append( 'a'+qk + ".jpg")
  # print 've' , len(k2b)
  plotkp([k2b], svgfn='k2ve.svg', x0=800, y0=600, poly = []) 

  return k2b,ddescb, tiles




def mosaicfeaturescentersabs(centers, z=16):
 
  mins = np.amin(centers, axis=0)
  ptps = np.ptp(centers,axis=0)

  minlon = mins[0]
  maxlat = mins[1]+ptps[1]

  # print minlon,maxlat
  tiles=[]
  txmin1, tymin1 =lonlat2xy(mins[0],mins[1]+ptps[1], z)
  txmax1, tymax1 =lonlat2xy(mins[0]+ptps[0],mins[1], z)


  # print txmax1-txmin1
  # print tymax1-tymin1

  qks=lonlat2qk(mins[0],mins[1]+ptps[1],z)
  qk0 = qk2qk(qks,-1,-3)
  wqk = txmax1-txmin1+4
  hqk = tymax1-tymin1+5

  qklist=[]

  for yy in range(hqk):
    for xx in range(wqk):
      qklist.append(qk2qk(qk0,xx,yy))
  poly=[]
  for qk in qklist:
    txq,tyq, z = qk2xy(qk)   
    tx = txq 
    ty = tyq 
    binb =  Config.vefeat + 'a'+qk + Config.ext + ".bin"

    ax, ay = 256*tx, 256*ty
    k, d = getfeat(binb, dx = ax, dy = ay)

    try:
      # fail on first iter
      ddescb = np.append(ddescb,d, axis=0)
      k2b.extend(k)
      minx=min(ax,minx)
      miny=min(ay,miny)
    except:
      # first iter
      binb =  Config.vefeat + 'a'+qk + Config.ext + ".bin"
      k2b, ddescb = getfeat(binb, dx = 0, dy = 0)
      minx=ax
      miny=ay
    rect=[[ax, ay],[ax+256, ay],[ax+256, ay+256],[ax, ay+256]]  
    poly.append(rect)        

    tiles.append( 'a'+qk + ".jpg")
  # print 've' , len(k2b)
  plotkp([k2b], svgfn='k2ve.svg', x0=-minx, y0=-miny, poly = poly,tiles=tiles) 
  return k2b,ddescb, tiles












def plotp1p2inliersabs(p1list=[],p2list=[],status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri=''):

  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">') 

  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  


  tx0, ty0, z=qk2xy(qklist[0] )
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0

      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';

 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )

  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.2);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]

  style2=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.4);stroke-width:2;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.4);stroke-width:2;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.4);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.4);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]  
  ns=0  

  nbad=0
  ngood=0
  
  x0=tx0*256
  y0=ty0*256
  nt=0
  for p1 in p1list:
      print 'p1len', len(p1)  
      nt+=1
      for n in range(len(p1)):
 
        x1 , y1 = p1[n].pt[0]-x0, p1[n].pt[1]-y0
 
        
        try:
          s=status[n]
        except:
          s=0  
        if s <1:
          nbad+=1
          with open(svgfn, mode='a') as ff:
            # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
            # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="'+style[1]+'"  />' )
            ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[nt]+'"  />' )
            # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
          continue
        ngood += 1  

        with open(svgfn, mode='a') as ff:

          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+11)+'" y2="'+str(y1+3)+'"  style="'+style2[nt]+'" />' )
          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+3)+'" y2="'+str(y1+11)+'"  style="'+style2[nt]+'"   />' )
          # ff.write('<line x1="'+str(x2)+'" y1="'+str(y2)+'" x2="'+str(x2-11)+'" y2="'+str(y2-3)+'" style="stroke:rgba(20,255,120,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )
          # ff.write('<line x1="'+str(x2)+'" y1="'+str(y2)+'" x2="'+str(x2-3)+'" y2="'+str(y2-11)+'" style="stroke:rgba(20,255,120,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )

  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:

    
    for cc in range(len(corn)):
      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-x0, cnow[1]-y0
      x2 , y2 = cnext[0]-x0, cnext[1]-y0
      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc+1)+';'
      s3=linetype[nc%len(linetype)]
  
      with open(svgfn, mode='a') as ff:
        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />' )
    nc+=1
  svgout ='<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>';
  with open(svgfn, mode='a') as file:
      file.write(svgout) 


  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  # print 'outsvg',ngood,nbad, svgfn 
  # print 'outsvg imuri',imuri



 




# def plotquads( corners= [], cornstyle={} ,svgfn='kpquads.svg' , tilesource= '/F5T/oc/Flight-Imagery/work/ve/',zoom=18  ,scale=1.0 ):
def plotquads(  cornstyle={} , svgfn='kpquads.svg' , tilesource= '/F5T/oc/Flight-Imagery/work/ve/',zoom=18  ,scale=1.0 ):

  # print 'plotquads', cor
  svgout = ''
  try:
    limits = cornstyle['limits']['cc'] 
  except:
    limits=oc.qklistbounds(['0230102'])

  qklist = corna2qklist([limits],zoom)
  # print len(qklist)

  tx0, ty0, z=qk2xy(qklist[0] )
  px0 = tx0 * 256
  py0 = ty0 * 256
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale('+str(scale)+')">') 

  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0
 
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="http://t0.tiles.virtualearth.net/tiles/a'+qk+'.jpeg?g=1398" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="'+tilesource+'a'+qk+'.jpg" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
  
  with open(svgfn, mode='a') as ff:
    ff.write(svgout )
  
  nc=0    
  # for c2 in corners:
  #   corn = cornlonlat2pxpy(c2,zoom)
  #   for cc in range(len(corn)):
  #     cnext = corn[(cc+1)%len(corn)]
  #     cnow = corn[cc]
  #     x1 , y1 = cnow[0]-px0, cnow[1]-py0
  #     x2 , y2 = cnext[0]-px0, cnext[1]-py0
  #     s3='stroke:rgba(255,220,0,0.7); stroke-width:1;'
     
  
  #     with open(svgfn, mode='a') as ff:
  #       ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />\n' )
  #   nc+=1
  

  for c0 in cornstyle:

    
    corn = cornlonlat2pxpy(cornstyle[c0]['cc'],zoom)
    try:
      s3 = cornstyle[c0]['style']
    except:
      s3='stroke:rgba(0,0,255,0.7);fill:rgba(0,255,255,0.0); stroke-width:2;'
    if c0=='limits':
      s3 = 'stroke:rgba(0,0,255,0.7);fill:rgba(0,255,255,0.2); stroke-width:2;'
              
    polyp=''
    for cc in range(len(corn)):
      # print 'cc',cc
      # cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-px0, cnow[1]-py0
      # x2 , y2 = cnext[0]-px0, cnext[1]-py0
      polyp +=" "+str(x1)+','+str(y1)
       
  
    with open(svgfn, mode='a') as ff:
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />\n' )
      ff.write('<polygon points="'+polyp+'"  style="'+s3+';"  />\n')

 

  with open(svgfn, mode='a') as file:
      file.write('</g></svg>')

  coutb('wrote '+svgfn)    



 

def plotkpnew(kplist=[] ,svgfn='kpstatus.svg' ,scale=1.0, qklist=[]  ):

  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale('+str(scale)+')">') 

  svgout = ''
  #
  try:
    x0, y0, z=qk2pxpy(qklist[0] )
  except:
    x0 , y0 = kplist[0][0].pt[0], kplist[0][0].pt[1]  
  print 'plotkpnew qklist', len(qklist)

  for qk in qklist:

      px, py, z=qk2pxpy(qk )
 
 
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(px-x0)+'px" y="'+str(py-y0)+'px" height="256px" width="256px" style=" opacity:1;"/>';
 
  
  with open(svgfn, mode='a') as ff:
    ff.write(svgout )

  for p1 in kplist:
    print 'plotkpnew', len(p1) 
    for n in range(len(p1)):
      x1 , y1 = p1[n].pt[0]-x0 , p1[n].pt[1]-y0 

      with open(svgfn, mode='a') as ff:

        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+5)+'" y2="'+str(y1+3)+'"  style="stroke:rgba(255,222,0,0.8); stroke-width:1;" />' +'\n')
        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+3)+'" y2="'+str(y1+5)+'"  style="stroke:rgba(255,222,0,0.8); stroke-width:1;"   />' )

 
  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
 



def drawmatches(kp1 ,kp2, matches, svgfn='kpstatus.svg' ,scale=1.0  ):

  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale('+str(scale)+')">') 

  svgout = ''
  x0 , y0 = kplist[0][0].pt[0], kplist[0][0].pt[1]   
  for p1 in kplist:
      for n in range(len(p1)):
        x1 , y1 = p1[n].pt[0]-x0+1000, p1[n].pt[1]-y0+1000
 
        with open(svgfn, mode='a') as ff:

          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+11)+'" y2="'+str(y1+3)+'"  style="stroke:rgba(255,222,0,0.8); stroke-width:1;" />' )
          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+3)+'" y2="'+str(y1+11)+'"  style="stroke:rgba(255,222,0,0.8); stroke-width:1;"   />' )
 
 
  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
 





def plotkpinliersabs(kp1list=[] ,status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri=''):

  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">') 

  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  


  tx0, ty0, z=qk2xy(qklist[0] )
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0

      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';

 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )

  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.2);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]

  style2=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.4);stroke-width:2;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.4);stroke-width:2;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.4);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.4);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]  
  ns=0  

  nbad=0
  ngood=0
  
  x0=tx0*256
  y0=ty0*256
  nt=0
  print 'plotkpinliersabs kp1listlen', len(kp1list)  
  for p1 in kp1list:
      print 'plotkpinliersabs p1len', len(p1)  
      nt+=1
      for n in range(len(p1)):
 
        x1 , y1 = p1[n].pt[0]-x0, p1[n].pt[1]-y0
 
        
        try:
          s=status[n]
        except:
          s=0  
        if s <1:
          nbad+=1
          with open(svgfn, mode='a') as ff:
            # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
            # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="'+style[1]+'"  />' )
            ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[nt]+'"  /> \n' )
            # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
          continue
        ngood += 1  

        with open(svgfn, mode='a') as ff:

          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+11)+'" y2="'+str(y1+3)+'"  style="stroke:rgba(220,255,120,0.8);stroke-width:2;" />' )
          ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+3)+'" y2="'+str(y1+11)+'"  style="stroke:rgba(220,255,120,0.8);stroke-width:2;"  />\n' )
 

  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:

    
    for cc in range(len(corn)):
      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-x0, cnow[1]-y0
      x2 , y2 = cnext[0]-x0, cnext[1]-y0
      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc+1)+';'
      s3=linetype[nc%len(linetype)]
  
      with open(svgfn, mode='a') as ff:
        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />' )
    nc+=1
  svgout ='<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>';
  with open(svgfn, mode='a') as file:
      file.write(svgout) 


  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  # print 'outsvg',ngood,nbad, svgfn 
  # print 'outsvg imuri',imuri






def makecluster(imuri,ncluster):

  imgolist = {}
  k2a=[]
  centers=[]
  corners=[]
 
  Hcumu = np.eye(3, dtype=float)
 
  for nc in range(ncluster):
    imgolist[imuri] = getimo(imuri)
    imurib = imgolist[imuri]['relabdone']['imurib']
    
    bina =  Config.processed + imuri + "/feats.bin"
    k2a0, ddesca1 = getfeat(bina) 
    print 'k2a0', len(k2a0)


    cornersp = np.plot(imgolist[imuri]['relabdone']['orig'])

    cornabp = np.float32(imgolist[imuri]['relabdone']['coords'])
    Hinc = cv2.getPerspectiveTransform( cornersp, cornabp)
    k2a1t = xformkp(k2a0, Hcumu)
    k2a.extend(k2a1t)
    centers.append((float(imgolist[imuri]['ocguide']['lon']), float(imgolist[imuri]['ocguide']['lat'])))
    c2 = np.float32(cv2.perspectiveTransform(cornersp.reshape(1, -1, 2), Hcumu).reshape(-1, 2) )
    corners.append(c2)
    Hcumu = np.dot(Hcumu,Hinc)

    imuri = imurib
    # print Hcumu
    try:
      ddesca = np.append(ddesca,ddesca1, axis=0)
    except:
      ddesca = ddesca1
 
          
  # print 'k2a',  len(k2a)  
  plotkp([k2a], svgfn='k2a.svg', x0=800, y0=1600, poly = [])    
  return imgolist, k2a, ddesca, centers, corners  













def kp2p(kpcv):
  """
  convert cv2 keypoint list to  point xy
  """
  pkp = np.empty([len(kpcv),2])
  for n in range(len(kpcv)):
    # for p in k2a:     
    kp=kpcv[n]
    pkp[n] = [kp.pt[0], kp.pt[1]]
  return pkp  






def kp2np(kpcv):
  """
  convert cv2 keypoint list to  6 wide float numpy keypoints
  """
  pkp = np.empty([len(kpcv),6])
  for n in range(len(kpcv)):
    # for p in k2a:     
    kp=kpcv[n]
    pkp[n] = [kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
  return pkp  


def np2kp(knp):
  """
  convert  6 wide float numpy keypoints tp cv2 keypoint list 
  """

  kp=[]
  for k1 in knp:
    kp.append(cv2.KeyPoint(k1[0], k1[1], k1[2],  k1[3],  k1[4], int(k1[5]) ) )

  return kp  
 
  




def plotvepic(qklist=[], svgfn='kpvepic.svg'):
  svgout=''


  qklist0=[]
  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  
  tx0, ty0, z= qk2xy(qklist[0] )  
  for qk in qklist:

      tx, ty, z= qk2xy(qk )
      dxa=tx
      dya=ty
      # print qk
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str( (dxa-tx0)*256)+'px" y="'+str((dya-ty0)*256)+'px" height="256px" width="256px" style=" opacity:1;"   dxa="'+str(dxa)+'" />';
 

  svgo='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform=" translate(0,0)">' + svgout + '</g></svg>'
  with open(svgfn, mode='w') as ff:
        ff.write(svgo )    









def cornlonlat2pxpy(corn,z):
    out =[]
    for n in range(len(corn)):
        px,py = lonlat2pxpy(corn[n][0], corn[n][1], z)
        out.append([px,py])
    return out    


def cornpxpy2lonlat(corn,z):
    out =[]
    for n in range(len(corn)):
        px,py = pxpy2lonlat(corn[n][0], corn[n][1], z)
        out.append([px,py])
    return out    

 

def autoplotxy(corn, svgfn ='autoplot.svg'):

  out=''
 

  for cc in  corn :
    try:
      minx=  min(cc[0],minx)
      miny=  min(cc[1],miny)        
      maxx=  max(cc[0],maxx)
      maxy=  max(cc[1],maxy)
    except:
      minx= cc[0] 
      miny= cc[1] 
      maxx= cc[0] 
      maxy= cc[1] 

  poly = ' '
  xr = max(maxx-minx, maxy-miny)
  scale= (800/xr)
  for cc in  corn :

 

    poly += ' '+str((cc[0]-minx)*scale+100)+','+str((cc[1]-miny)*scale+100)




 
  with open(svgfn, mode='w') as ff:        
    ff.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform=" ">')     
    ff.write('<polygon points="'+poly+'"  style="fill: rgba(255,100,0,0.1) ; stroke:rgba(255,100,0,0.8) ;stroke-width:2;"  />\n\n' )
 
    ff.write('</g></svg>' )



# def plotkpinliersv2(p1, p2, status, x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri='', abscoords=False, info='plotkpinliers'):

def triangle(x=0.0):
  return max(1.0 - abs(x),0)


# def rainbow(n,steps=0, opacity=1.0):
#   x= (n %steps)/steps *3
#   r=255* (triangle(x ) +  triangle(x -3.0))
#   g=255*triangle(x-1.0)
#   b=255*triangle(x-2.0)
#   return 'rgba('+"%0d"%r+','+"%0d"%g+','+"%0d"%b+', '+str(opacity)+')'


def rainbow(nf, opacity=1.0):
  x= (nf*3)%3
  r=255* (triangle(x ) +  triangle(x -3.0))
  g=255*triangle(x-1.0)
  b=255*triangle(x-2.0)
  return 'rgba('+"%0d"%r+','+"%0d"%g+','+"%0d"%b+', '+str(opacity)+')'





def plotkpinliersv1(kp1list=[] ,status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri='', info="plotkpinliersv2"):
  # print 'plotkpinliersv1' 
  # print len(kp1list[0]),len(kp1list[1]),len(status)
 
  xo,yo, z = qk2pxpy(qklist[0])
 
  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  
  tx0, ty0, z=qk2xy(qklist[0] )
  # x0=tx0*256
  # y0=ty0*256
  x0=0
  y0=0
 
  for qk in qklist:
      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256-x0)+'px" y="'+str(dya*256-y0)+'px" height="256px" width="256px" style=" opacity:0.9;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';

  # with open(svgfn, mode='w') as file:
  #     file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate('+str(-xo)+','+str(-yo)+')">')   
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate(0,0)">') 
 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )

  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:
    # print 'plotkpinliersv1 corn'
    # print  corn

    poly='  '
    polypx=' '
    for cc in range(len(corn)):
 
      x1g, y1g = lonlat2pxpy(corn[cc][0], corn[cc][1], z)
 
      poly += ' '+str(x1g)+','+str(y1g)
      polypx += ' '+str(x1g-xo)+','+str(y1g-yo)
      # print poly
 
    with open(svgfn, mode='a') as ff:
        ff.write('<polygon points="'+poly+'"  style="fill: rgba(255,100,0,0.1) ; stroke:rgba(255,100,0,0.8) ;stroke-width:2;"  />\n\n' )
        ff.write('<polygon points="'+polypx+'"  style="fill: rgba(255,100,0,0.1) ; stroke:rgba(255,100,0,0.8) ;stroke-width:2;"  />\n\n' )
    nc+=1
 
 

 

  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.6);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]
  ns=0  

  nbad=0
  ngood=0
  
 
  try:
    p1=kp2p(kp1list[0])
    p2=kp2p(kp1list[1])
  except:
    p1= (kp1list[0])
    p2= (kp1list[1])
      
  for n in range(len(status)):
    x1 , y1 = p1[n][0]-x0, p1[n][1]-y0
    x2 , y2 = p2[n][0]-x0, p2[n][1]-y0

    adx = p2[n][0] - p1[n][0]
    ady = p2[n][1] - p1[n][1]
 
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
  
    if status[n] <1:
      nbad+=1
      with open(svgfn, mode='a') as ff:
        # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="stroke:rgba(255,222,0,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="stroke:rgba(255,0,222,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
   
      continue
    ngood += 1  
    err= ((x1-x2)**2+(y1-y2)**2)**0.5
    if err>100: err=6
    e4=12
    e3 =8
    e2=4
 
    aa=''   
   
    aa+='<circle cx="'+str((x1+x2)/2)+'" cy="'+str((y1+y2)/2)+'" r="'+str(err)+'" style="stroke:rgba(220,220,0,1.0); fill:rgba(222,0,243,1.0);stroke-width:1;"  />'  
    aa+='<line x1="'+str(x1+e2)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2-e4)+'" style="stroke:rgba(20,255,120,0.8);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(20,255,120,0.8); stroke-width:2;"  />'  
  

    xc = (ngood*8)%800+100
    yc = 25+math.floor((ngood*8)/800)*25
    ye=yc-err

 
    aa+='<circle cx="'+str(xc )+'" cy="'+str(yc )+'" r="'+str(err)+'" style="stroke:rgba(220,222,0,1.0); fill:'+rainbow(24-err)+';stroke-width:1;"  />'
 

    with open(svgfn, mode='a') as ff:
        ff.write(aa)
  

  footer= '</g>'+'<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>'+'<g transform="translate(0,0)"><text x="10"  y="20" style="font-family: Helvetica; font-size:16;  stroke: rgba(220,110,0,0.0);fill:rgba(0,220,180,0.8);">'+info+'</text>'+'</g></svg>'  
  with open(svgfn, mode='a') as file:
      file.write(footer) 
  print 'oc.plotkpinliers',ngood,nbad, svgfn 



























def plotkpinliersv2(kp1list=[] ,status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri=''):
  p1= (kp1list[0])
  p2= (kp1list[1])
  tx0, ty0, z=oc.qk2xy(qklist[0] )
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale(1.0)">')
      file.write('<g transform="translate('+str(-tx0*256)+','+str(-ty0*256)+')">') 

  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  



  tx0, ty0=0,0
  for qk in qklist:

      tx, ty, z=oc.qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0

      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';

 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )


  for corn in corners:

    
    for cc in range(len(corn)):
      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-x0, cnow[1]-y0
      x2 , y2 = cnext[0]-x0, cnext[1]-y0
      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc+1)+';'
      s3=linetype[nc%len(linetype)]

      with open(svgfn, mode='a') as ff:
        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />\n' )
    nc+=1
  svgout ='<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>';
  with open(svgfn, mode='a') as file:
      file.write(svgout) 












  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.6);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]
  ns=0  

  nbad=0
  ngood=0
  
  x0=0
  y0=0
  for n in range(len(p1)):
    x1 , y1 = p1[n][0]-x0, p1[n][1]-y0
    x2 , y2 = p2[n][0]-x0, p2[n][1]-y0

    adx = p2[n][0] - p1[n][0]
    ady = p2[n][1] - p1[n][1]
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
  
    if status[n] <1:
      nbad+=1
      with open(svgfn, mode='a') as ff:
        # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="'+style[1]+'"  />' )
        ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
   
      continue
    ngood += 1  
#   # cv2.line(vis, (x1, y1), (x2, y2), green)
    with open(svgfn, mode='a') as ff:
 
      # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="5" style="'+style[0]+'"  />' )
      # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="3" style="'+style[2]+'"  />' )
      
      ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+11)+'" y2="'+str(y1+3)+'" style="stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )
      ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+3)+'" y2="'+str(y1+11)+'" style="stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )
      ff.write('<line x1="'+str(x2)+'" y1="'+str(y2)+'" x2="'+str(x2-11)+'" y2="'+str(y2-3)+'" style="stroke:rgba(20,255,120,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )
      ff.write('<line x1="'+str(x2)+'" y1="'+str(y2)+'" x2="'+str(x2-3)+'" y2="'+str(y2-11)+'" style="stroke:rgba(20,255,120,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1 +adx )+'" y2="'+str(y1)+'" style="'+s3+'"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+adx )+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="100" y1="100" x2="'+str(100+p1[n][0]- p2[n][0])+'" y2="'+str(100+p1[n][1]- p2[n][1])+'" style="'+style[2]+'"  />' )
  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
 


  with open(svgfn, mode='a') as file:
      file.write('</g></g></svg>') 
  print 'outsvg',ngood,nbad, svgfn 
  print 'outsvg imuri',imuri


















def plotkpinliersv1(kp1list=[] ,status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri='', info="plotkpinliersv2", scale=1.0):
  # print 'plotkpinliersv1' 
  # print len(kp1list[0]),len(kp1list[1]),len(status)
 
  xo,yo, z = qk2pxpy(qklist[0])
 
  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  
  tx0, ty0, z=qk2xy(qklist[0] )
  # x0=tx0*256
  # y0=ty0*256
  x0=0
  y0=0
 
  for qk in qklist:
      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256-x0)+'px" y="'+str(dya*256-y0)+'px" height="256px" width="256px" style=" opacity:0.9;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';

  # with open(svgfn, mode='w') as file:
  #     file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate('+str(-xo)+','+str(-yo)+')">')   
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="scale('+str(scale)+')"><g transform="translate(0,0)">') 
 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )

  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:
    # print 'plotkpinliersv1 corn'
    # print  corn

    poly='  '
    polypx=' '
    for cc in range(len(corn)):
 
      x1g, y1g = lonlat2pxpy(corn[cc][0], corn[cc][1], z)
 
      poly += ' '+str(x1g)+','+str(y1g)
      polypx += ' '+str(x1g-xo)+','+str(y1g-yo)
      # print poly
 
    with open(svgfn, mode='a') as ff:
        ff.write('<polygon points="'+poly+'"  style="fill: rgba(255,100,0,0.1) ; stroke:rgba(255,100,0,0.8) ;stroke-width:2;"  />\n\n' )
        ff.write('<polygon points="'+polypx+'"  style="fill: rgba(255,100,0,0.1) ; stroke:rgba(255,100,0,0.8) ;stroke-width:2;"  />\n\n' )
    nc+=1
 
 

 

  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.6);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]
  ns=0  

  nbad=0
  ngood=0
  
 
  try:
    p1=kp2p(kp1list[0])
    p2=kp2p(kp1list[1])
  except:
    p1= (kp1list[0])
    p2= (kp1list[1])
      
  for n in range(len(status)):
    x1 , y1 = p1[n][0]-x0, p1[n][1]-y0
    x2 , y2 = p2[n][0]-x0, p2[n][1]-y0

    adx = p2[n][0] - p1[n][0]
    ady = p2[n][1] - p1[n][1]
 
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
  
    if status[n] <1:
      nbad+=1
      with open(svgfn, mode='a') as ff:
        # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="stroke:rgba(255,222,0,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="stroke:rgba(255,0,222,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
   
      continue
    ngood += 1  
    err= ((x1-x2)**2+(y1-y2)**2)**0.5
    if err>100: err=6
    e4=12
    e3 =8
    e2=4
 
    aa=''   
   
    aa+='<circle cx="'+str((x1+x2)/2)+'" cy="'+str((y1+y2)/2)+'" r="'+str(err)+'" style="stroke:rgba(220,220,0,1.0); fill:rgba(222,0,243,1.0);stroke-width:1;"  />'  
    aa+='<line x1="'+str(x1+e2)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2-e4)+'" style="stroke:rgba(20,255,120,0.8);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(20,255,120,0.8); stroke-width:2;"  />'  
  

    xc = (ngood*8)%800+100
    yc = 25+math.floor((ngood*8)/800)*25
    ye=yc-err

 
    aa+='<circle cx="'+str(xc )+'" cy="'+str(yc )+'" r="'+str(err)+'" style="stroke:rgba(220,222,0,1.0); fill:'+rainbow(24-err)+';stroke-width:1;"  />'
 

    with open(svgfn, mode='a') as ff:
        ff.write(aa)
  

  footer= '</g>'+'<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>'+'<g transform="translate(0,0)"><text x="10"  y="20" style="font-family: Helvetica; font-size:16;  stroke: rgba(220,110,0,0.0);fill:rgba(0,220,180,0.8);">'+info+'</text>'+'</g></g></svg>'  
  with open(svgfn, mode='a') as file:
      file.write(footer) 
  print 'oc.plotkpinliers',ngood,nbad, svgfn 


































def plotkpinliers(p1, p2, status, x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri='', abscoords=False, info='plotkpinliers'):
  print 'plotkpinliers',imuri
  xo,yo=0,0  
  if abscoords:
    xo,yo, z = qk2pxpy(qklist[0])




  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  


  tx0, ty0, z=qk2xy(qklist[0] )
  if abscoords:
    tx0, ty0 = 0, 0
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx-tx0
      dya=ty-ty0

      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';


  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate('+str(-xo)+','+str(-yo)+')">') 
 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )

  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.6);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]
  ns=0  

  nbad=0
  ngood=0
  
  x0=0
  y0=0
  for n in range(len(p1)):
    x1 , y1 = p1[n][0]-x0, p1[n][1]-y0
    x2 , y2 = p2[n][0]-x0, p2[n][1]-y0

    adx = p2[n][0] - p1[n][0]
    ady = p2[n][1] - p1[n][1]
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
  
    if status[n] <1:
      nbad+=1
      with open(svgfn, mode='a') as ff:
        # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="stroke:rgba(255,222,0,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="stroke:rgba(255,0,222,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
   
      continue
    ngood += 1  
    err= ((x1-x2)**2+(y1-y2)**2)**0.5
    e4=12
    e3 =8
    e2=4
#   # cv2.line(vis, (x1, y1), (x2, y2), green)

    aa=''   
    aa+='<circle cx="'+str((x1+x2)/2)+'" cy="'+str((y1+y2)/2)+'" r="'+str(err)+'" style="stroke:rgba(220,0,220,0.0); fill:rgba(0,240,243,0.6);stroke-width:0;"  />'  
    aa+='<line x1="'+str(x1+e2)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' 
    aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' 
    # aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,110,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2-e4)+'" style="stroke:rgba(20,255,120,0.8);stroke-width:1;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(20,255,120,0.8); stroke-width:1;"  />'  
    # aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(222,222,220,0.8); stroke-width:4;"  />'  

    xc = ngood*2+100
    yc =100
    ye=yc-err
 
    aa+='<circle cx="'+str(xc+xo)+'" cy="'+str(yc+yo)+'" r="'+str(err)+'" style="stroke:rgba(220,222,0,1); fill:rgba(220,0,123,0.6);stroke-width:1;"  />'

    with open(svgfn, mode='a') as ff:
        ff.write(aa)
 
      # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="5" style="'+style[0]+'"  />' )
      # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="3" style="'+style[2]+'"  />' )
      


      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+adx )+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="100" y1="100" x2="'+str(100+p1[n][0]- p2[n][0])+'" y2="'+str(100+p1[n][1]- p2[n][1])+'" style="'+style[2]+'"  />' )
  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:

    
    for cc in range(len(corn)):
      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-x0, cnow[1]-y0
      x2 , y2 = cnext[0]-x0, cnext[1]-y0
      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc+1)+';'
      s3=linetype[nc%len(linetype)]
      aa=''
      x1g, y1g = lonlat2pxpy(cnow[0], cnow[1], z)
      x2g, y2g = lonlat2pxpy(cnext[0], cnext[1], z)
      aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />'
      aa+='<line x1="'+str(x1g)+'" y1="'+str(y1g)+'" x2="'+str(x2g)+'" y2="'+str(y2g)+'" style="'+s3+'"  />'
      with open(svgfn, mode='a') as ff:
        ff.write(aa )
    nc+=1
 
 

  footer= '</g>'+'<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>'+'<g transform="translate(0,0)"><text x="10"  y="20" style="font-family: Helvetica; font-size:16;  stroke: rgba(220,110,0,0.0);fill:rgba(0,220,180,0.8);">'+info+'</text>'+'</g></svg>'  
  with open(svgfn, mode='a') as file:
      file.write(footer) 
  print 'oc.plotkpinliersv1 ',ngood,nbad, svgfn  ,imuri











 


def kpfilterpoly(k2a, ddesca, poly=[]):
  # return k2a, ddesca
  k2af = []
  # ddescaf =  np.empty([1, 61], dtype=np.uint8)
 
  # print ddesca.shape 
  mask = np.ones(len(k2a), dtype=bool)
  for n in range(len(k2a)):
    x,y = k2a[n].pt[0],k2a[n].pt[1]
    k=k2a[n]
    mask[n] = False
    for npoly in range(len(poly)):
      if  oc.point_inside_polygon(x, y, poly[npoly]):
        
        mask[n] = True
        continue
    if mask[n]:
        k2af.append(k)   
       
  ddescaf = ddesca[mask,...]
  print 'kpfilterpoly ddescaf.shape', ddesca.shape, 'to',ddescaf.shape 
  print 'kpfilterpoly len(k2a)',len(k2a),'to', len(k2af),  k2af[0].pt
 
  return k2af, ddescaf



def kpfilter(k2a, ddesca):
  # return k2a, ddesca
  k2af = []
  # ddescaf =  np.empty([1, 61], dtype=np.uint8)
 
  print ddesca.shape 
 
  mask = np.ones(len(k2a), dtype=bool)
  for n in range(len(k2a)):
    x,y = k2a[n].pt[0],k2a[n].pt[1]
    if x < 300 or x > 600:
      

      k2af.append(k2a[n])
      continue
    mask[n] = False   
 
        
  ddescaf = ddesca[mask,...]
  print ddescaf.shape 
 
  return k2af, ddescaf

def cout(data ):
  print(  data  )
def coutb(data ):
  print(Fore.WHITE  + Back.BLUE + data + Style.RESET_ALL)
def coutr(data ):
  print(Fore.WHITE  + Back.RED + data + Style.RESET_ALL)
def coutg(data ):
  print(Fore.BLACK  + Back.GREEN + data + Style.RESET_ALL)
def couty(data ):
  print(Fore.BLACK  + Back.YELLOW + data + Style.RESET_ALL)
 

def coordspx2lonlat(clist, zoom=18):
  latc = []
  lonc = []
  for n2 in range(4):
    latcc, loncc =  Pixel2LL(float(clist[n2][0] ),float(clist[n2][1] ), zoom)
    latc.append(latcc)
    lonc.append(loncc)
    try:
      lonlat.append([loncc, latcc])
    except:
      lonlat= [[loncc, latcc]]
  return lonlat    
      


def coordslonlat2pxpy(clist, zoom=18):
  pxpy = []
 
  for n2 in range(4):
    px, py =  lonlat2pxpy(float(clist[n2][0] ),float(clist[n2][1] ), zoom)
    pxpy.append([px,py])
 
  return pxpy   
   



def plotkps(kp1, kp2=[], status=[],  x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [[]], imuri='',scale=1.0):

  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2400" height="2200"><g transform="scale('+str(scale)+')">') 
  dx=0
  dy=0
  svgout = ''
  tx0, ty0, z=oc.qk2xy(qklist[0] )

  svgout ='<g transform="translate(0,0)"><image xlink:href="'+imuri+'" x="0px" y="0px"  height="256px" width="256px"  style=" opacity:1;"/></g>';
  with open(svgfn, mode='a') as file:
      file.write(svgout) 

 
  ns=0  

  nbad=0
  ngood=0
  

  for n in range(len(kp1)):
   
    x1 , y1 = kp1[n].pt[0]+x0, kp1[n].pt[1]+y0
    # x2 , y2 = kp2[n][0]+x0, kp2[n][1]+y0   

    sa='stroke:rgba(200,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:2;'
    sb='stroke:rgba(0,0,220,0.2); fill:rgba(0,100,220,0.8);stroke-width:2;'
    sline='stroke:rgba(0,220,120,0.2); fill:rgba(255,100,0,0.8);stroke-width:2;'
    r='2'
    rb='3'
#   # cv2.line(vis, (x1, y1), (x2, y2), green)
    # if len(status) == len(p1):
    try:  
      if status[0][n] >0 :
        sa='stroke:rgba(200,10,0,0.6); fill:rgba(255,100,0,0.2);stroke-width:2;'
        sb='stroke:rgba(0,0,200,0.6); fill:rgba(250,110,0,0.5);stroke-width:2;'
        sline='stroke:rgba(0,10,220,0.6); fill:rgba(255,100,0,0.2);stroke-width:2;'
        r='4'
        rb='6'

      if  status[1][n] >0 :
        sa='stroke:rgba(200,10,0,0.6); fill:rgba(255,100,0,0.2);stroke-width:2;'
        sb='stroke:rgba(0,0,200,0.6); fill:rgba(250,110,0,0.5);stroke-width:2;'
        sline='stroke:rgba(220,10,0,0.4); fill:rgba(255,100,0,0.2);stroke-width:6;'
        r='4'
        rb='6'
    except:
      try:
        if status[n] >0 :
          sa='stroke:rgba(200,10,0,0.6); fill:rgba(255,100,0,0.2);stroke-width:2;'
          sb='stroke:rgba(0,0,200,0.6); fill:rgba(250,110,0,0.5);stroke-width:2;'
          sline='stroke:rgba(0,10,220,0.6); fill:rgba(255,100,0,0.2);stroke-width:2;'
          r='4'
          rb='6'
      except:
        pass   


#   # cv2.line(vis, (x1, y1), (x2, y2), green)

   
    with open(svgfn, mode='a') as ff:
      ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="'+r+'" style="'+sa+'"  />' )
      # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="'+rb+'" style="'+sb+'"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2 )+'" y2="'+str(y2 )+'" style="'+sline+'"  />' )



  nc=0   

  sc=['stroke:rgba(255,220,0,0.7); stroke-width:2px;',
  'stroke:rgba(210,110,0,0.4); stroke-width:12px;',
  'stroke:rgba(0,0,220,0.9); stroke-width:2px;',
  'stroke:rgba(220,0,0,0.9); stroke-width:3px;']
  for cr in range(len(corners)):
    corn= corners[cr]
    # print "corn", corn

    for cc in range(len(corn)):

      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      # print cnow
      x1 , y1 = cnow[0]+x0, cnow[1]+y0
      x2 , y2 = cnext[0]+x0, cnext[1]+y0
      # print x1, y1, x2, y2 

      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc*2+2)+';'

  
      with open(svgfn, mode='a') as ff:
        ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+sc[nc]+'"  />' )
        
    nc+=1 

  with open(svgfn, mode='a') as file:
      file.write('</g></svg>') 
  print 'outsvg',ngood,nbad, svgfn 
  print 'outsvg imuri',imuri



 

def mosaicfeatures(qklist,vefeat = 'http://10.0.0.4/F5T/oc/Flight-Imagery/work/vefeat/', ext= ".jpg.bin" , abscoords=False):
  tiles=[]
  txmin, tymin ,tz=oc.qk2xy(qklist[0])
  if abscoords:
    txmin, tymin = 0, 0

  # print 'txmin, tymin',  txmin, tymin

  for qk in qklist:
      txq,tyq, z = oc.qk2xy(qk)   
      tx = txq - txmin 
      ty = tyq - tymin 
      binb =   vefeat + 'a'+qk +  ext 

      k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)
       


      try:
        # fail on first iter
        ddescb = np.append(ddescb,d, axis=0)
        k2b.extend(k)
      except:
        k2b, ddescb = getfeat(binb, dx = 256*tx, dy = 256*ty)
      tiles.append( 'a'+qk + ".jpg")


  return k2b,ddescb

def montagefeatures(qklist,tilesource = 'http://10.0.0.4/F5T/oc/Flight-Imagery/work/ve/', ext='.jpg', montagesave=""):

  print 'montagefeatures qklist', len(qklist),'tiles'

  img = oc.qkmontage(qklist, ve = tilesource )
  
  if len(montagesave)>0:
    img.save(montagesave,"JPEG")
  print 'montagefeatures saved', montagesave
 



  imgcv2=np.array(img)
  
  detector = cv2.AKAZE_create()
  detector.setThreshold(0.0005)
  return detector.detectAndCompute(imgcv2, None)


  tiles=[]
  txmin, tymin ,tz=oc.qk2xy(qklist[0])
  # descall =np.empty(120000, 61], dtype=np.uint8)
  # kpall = []

  kpfeat ={}
  sumkp=0
  for qk in qklist:      
      txq,tyq, z = oc.qk2xy(qk)   
      tx = txq - txmin 
      ty = tyq - tymin 
      imurl =   tilesource + 'a'+qk +  ext 
      kpfeat[qk]  = geturlfeat(imurl)
      sumkp+=len(kpfeat[qk]['kp'])
      print qk, len(kpfeat[qk]['kp'])

      try:
        # fail on first iter
        ddescb = np.append(ddescb,kpfeat[qk]['desc'], axis=0)
        k2b.extend(kpfeat[qk]['kp'])
      except:
        ddescb=kpfeat[qk]['desc']
        k2b=kpfeat[qk]['kp']
        
  print 'mosaicfeaturestiles', sumkp
  return k2b, ddescb

def geturlfeat(imurl):
  r = requests.get(imurl)
  img = Image.open(StringIO(r.content))
  imgcv2=np.array(img)
   # dnc {'kp':kp, 'desc': desc , 'shape' : img1.shape}
  return   dnc(imgcv2 , scale=1.0, AKAZEThreshold=0.0005  )
     


def mosaictiles(qklist,veroot = '/home/oc/F5T/oc/Flight-Imagery/work/ve/', ext= ".jpg" ):
  tiles=[]
  txmin, tymin ,tz=oc.qk2xy(qklist[0])

  # print 'txmin, tymin',  txmin, tymin

  for qk in qklist:
      txq,tyq, z = oc.qk2xy(qk)   
      tx = txq - txmin 
      ty = tyq - tymin 
 
      
      if (veroot[:4]=="http"): 

        imurl =   veroot + 'a'+qk +  ext 
        print 'ocmosaictiles getting',imurl
        kpfeat  = geturlfeat(imurl)
        k,d  = kpfeat['kp'], kpfeat['desc']
      else:
        k, d = getfeattile(qk,veroot = veroot ,ext=ext, dx = 256*tx, dy = 256*ty)

        

      try:
        # fail on first iter
        ddescb = np.append(ddescb,d, axis=0)
        k2b.extend(k)
      except:
        k2b, ddescb = k,d
      tiles.append( 'a'+qk + ".jpg")

      # print qk, len(k), len(k2b)
  try:
    k=k2b
  except:
    return [], []      
  return k2b,ddescb




def getfeattile(qk='02301020333323123', veroot='/home/oc/F5T/oc/Flight-Imagery/work/ve/', ext='.jpg',dx=0, dy=0, AKAZEThreshold = 0.0005):
  
  fn = veroot+'a'+qk+ext
  if not os.path.isfile(fn):
    print 'getfeattile no image', fn
    return [], []
  img = cv2.imread(fn)
 
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)
  k1, desc1 = detector.detectAndCompute(img, None)
  k2a=[]
  for kp in k1:
    k2a.append(cv2.KeyPoint(kp.pt[0]+dx, kp.pt[1]+dy, kp.size,  kp.angle,  kp.response, kp.octave ) )

  return k2a, desc1     











def mosaicfeaturesabs(qklist , vefeat =   'http://10.0.0.4/F5T/oc/Flight-Imagery/work/vefeat/', ext= ".jpg.bin" ):
 
  tiles=[]

  for qk in qklist:
      txq,tyq, z = oc.qk2xy(qk)   
      tx = txq  
      ty = tyq  
      binb =   vefeat + 'a'+qk +  ext  

      k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)
      if len(k)<1: continue
      # print' tx, ty' ,qk, k[0].pt
      
      try:
        # fail on first iter
        ddescb = np.append(ddescb,d, axis=0)
        k2b.extend(k)
      except:
        k2b, ddescb = getfeat(binb, 256*tx, dy = 256*ty)
 
 
  return k2b,ddescb 



def mosaicfeatureslonlat(qklist , vefeat =   'http://10.0.0.4/F5T/oc/Flight-Imagery/work/vefeat/', ext= ".jpg.bin" ):
 
  corn0 =  [[0,0],[256,0],[256,256],[0,256] ] 
  ddescb= np.empty([0, 61], dtype=np.uint8)
  k2b=[]
  for qk in qklist:
      px0, py0, z =  qk2pxpy(qk)   
      px1, py1, z =  qk2pxpy(qk2qk(qk,1,1))  
      cornqk =  [[px0,py0],[px1,py0],[px1,py1],[px0,py1]]  
 
      k, d = getfeat(  vefeat + 'a'+qk +  ext , dx = 0, dy = 0)
      if len(k)<1: continue
      Hpx2lonlat = cv2.getPerspectiveTransform(np.float32(corn0), np.float32(cornpxpy2lonlat(cornqk,z) ))

      klonlat, limits = xformkp(k,Hpx2lonlat)
 

      ddescb = np.append(ddescb,d, axis=0)
      k2b.extend(klonlat)
 
  return k2b,ddescb 





def mosaicfeatureswh(qk,wx,wy,  vefeat =   'http://10.0.0.4/F5T/oc/Flight-Imagery/work/vefeat/', ext= ".jpg.bin" ):
 
  tiles=[]
  
  for ty in range(0,wy):
    for tx in range(0,wx):
      q = oc.qk2qk(qk,tx,ty)
      # binb =  Config.vefeat + 'a'+q + ".jpg.bin"
      # k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)
      
      binb =  vefeat + 'a'+q +  ext 
      k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)
      # print tx, ty, len(k), binb
      try:
        ddescb = np.append(ddescb,d, axis=0)
        k2b.extend(k)
      except:
        k2b, ddescb = getfeat(binb, dx = 0, dy = 0)
            
 
      tiles.append( 'a'+q + ".jpg")

  
  return k2b,ddescb, tiles

 




def mosaicfeaturesqklist(qklist,  vefeat =   'http://10.0.0.4/F5T/oc/Flight-Imagery/work/vefeat/', ext= ".jpg.bin" ):
 
  
  for qk in qklist:
 
      tx, ty, z = oc.qk2xy(qk)
      binb =  vefeat + 'a'+qk +  ext 
      k, d = getfeat(binb, dx = 256*tx, dy = 256*ty)
 
      try:
        ddescb = np.append(ddescb,d, axis=0)
        k2b.extend(k)
      except:
        k2b, ddescb = getfeat(binb, dx = 0, dy = 0)
 
  return k2b,ddescb 



def getfeatimuri(imuri, aifeat = 'http://10.0.0.4/F5T/oc/Flight-Imagery/work/processed/', aifs='/home/oc/F5T/oc/Flight-Imagery/work/processed/',dx=0,dy=0):
  bina =  aifeat + imuri + "/feats.bin"
  binafs =  aifs + imuri + "/feats.bin"
  featdeca=''

  try:
    with open(bina, mode='rb') as file:
      featdeca = file.read()
  except:  
    r = requests.get(bina, stream=True)


    if r.status_code == 200:
        # r.raw.decode_content = True
        featdeca = r.content
    if r.status_code != 200:
        print "getfeatimuri pbin not found", bina
        return [], [] 
 
  if len(featdeca) < 1:
    coutr('getfeatimuri too few feats'+str(len(featdeca)))
    coutr(bina)
    return [], []  

  k2a=[]
  sads = len(featdeca)/85
  ddesca=np.empty([sads, 61], dtype=np.uint8)
  # print 'nfeat',len(featdeca)/85.0
  n1=0
  for gn in range(len(featdeca)/85):
    g=featdeca[gn*85:(gn+1)*85]
    d1 = np.fromstring(g[-61:], dtype=np.uint8)
    ddesca[n1,:]=d1
    fmt = "f" * 6
    k1 = list(struct.unpack(fmt, g[:24]))
 
    try:
      k2a.append(cv2.KeyPoint(k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(0) ) )
      # k2a.append(cv2.KeyPoint(k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(k1[5]) ) )
    except:
      print 'bina prob', bina
      print   k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(k1[5]) 
      return [],[]
    n1+=1

  te1=tnow()
  print 'oc.getfeatimuri file open', len(k2a)
  # print 'getfeat:', te1-ts1 
  return k2a, ddesca      


def getfeat(bina, dx=0, dy=0):
 
  ts1=tnow()
 
  if bina[:4]=='http':
 
    r = requests.get(bina, stream=True)

    if r.status_code == 200:
        # r.raw.decode_content = True
        featdeca = r.content
    if r.status_code != 200:
        #print "can't find", bina
        dddd=0
 
 
  else:
    #print 'file open'
    with open(bina, mode='rb') as file:
        featdeca = file.read()

  te1=tnow()
  # print 'gettotal:', te1-ts1  
  k2a=[]
  try:
    sads = len(featdeca)/85
 
  except:
    featdeca=''
    sads = 0 

     
  ddesca=np.empty([sads, 61], dtype=np.uint8)
  # print 'nfeat',len(featdeca)/85.0
 
  n1=0
  for gn in range(len(featdeca)/85):
    g=featdeca[gn*85:(gn+1)*85]
    d1 = np.fromstring(g[-61:], dtype=np.uint8)
    ddesca[n1,:]=d1
    fmt = "f" * 6
    k1 = list(struct.unpack(fmt, g[:24]))
 
    try:
      k2a.append(cv2.KeyPoint(k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(0) ) )
      # k2a.append(cv2.KeyPoint(k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(k1[5]) ) )
    except:
      #print 'bina prob', bina
      print   k1[0]+dx, k1[1]+dy, k1[2],  k1[3],  k1[4], int(k1[5]) 
      quit()
    n1+=1

  te1=tnow()
  # print 'getfeat:', te1-ts1 
  return k2a, ddesca      










def savefeat(kp1, desc1, fn = 'feats.bin'):
  feats=''
  for nkp in range(len(kp1)):
      # print 'kp',len(kp1),(kp1[0].__str__),len(desc1[0])
      kp = kp1[nkp]
      float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
      keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
      feats += keyp+desc1[nkp].tostring()
 
 
  with open(fn, 'wb') as f:
      f.write(feats)

 
def findfeatures(im):
    """
    find features of PIL img
    """
    width, height = im.size
    img = np.array(im) 
    try:
      img = img[:, :, ::-1].copy() 
    except:
      return 0, '0' 

    detector = cv2.AKAZE_create()
    AKAZEThreshold = 0.001
    detector.setThreshold(AKAZEThreshold)

    kp1, desc1 = detector.detectAndCompute(img, None)
 
    # print len(kp1), "kp", timetaken , "s"
    # ['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'angle', 'class_id', 'octave', 'pt', 'response', 'size']
    feats=''

    n1=0
    for nkp in range(len(kp1)):
      # print 'kp',len(kp1),(kp1[0].__str__),len(desc1[0])
      kp = kp1[nkp]
      float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
      keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
      # if (n1==0): print float_array
      g=keyp+desc1[nkp].tostring()
   
      # desc1[nkp].tostring()
      ss=np.fromstring(g[-61:], dtype=np.uint8)
      feats += g
      n1+=1
      # feats+= float_array.tostring() +  desc1[nkp].tostring()
    return len(kp1), feats  


def findtiles(src, out ):
  names = os.listdir(src)
  print len(names),'vetiles'
  ts=tnow()
  ndone=0
  for fn in names:
    # if ndone>1000: break

    z = len(fn)-5
    if z !=16:
      continue
    ndone+=1 
    if ndone%1000==0: print ndone
    outfn= out+fn+'.bin'
    # print fn, fn[:-4]
    if os.path.isfile(outfn):
      continue
    im = Image.open(src+fn)

    kpn, feats=findfeatures(im)
 
    with open(outfn, 'wb') as f:
      f.write(feats)
    te=tnow()
    print ndone, te-ts, "kpn", kpn, len(feats)
    ts = te 
    
    # exit()


def dist_filter_matches(kp1, kp2, matches, dist=100):
    mkp1, mkp2 = [], []
    for m in matches:
 
        distx=abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])
        disty=abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])  

        if distx <dist and  disty <dist :      
          mkp1.append( kp1[m.queryIdx] )
          mkp2.append( kp2[m.trainIdx] )      

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    print "distfilter", len(kp1), 'to',len(p1)
    return p1, p2, kp_pairs

 

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def filter_matches2(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        mkp1.append( kp1[m.queryIdx] )
        mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

 









# defgeorefsingle(imuri, bf):
#   # for n in range(nimg): 
#   #   pass

#   imgo = getimo(imuri)
#   if not imgo['ocguide'] :
#     print "ocguide not found"
#     return
#   # print imgo['ocguide'], imgo['relabdone']
#   ncluster=1


#   if not imgo['relabdone'] :
#     return

#   if ncluster<3:
#     return
 

#   tsdir=tnow()

#   bina =  Config.processed + imgo['imuri'] + "/feats.bin"

#   imapath = Config.processed + imgo['imuri'] + "/small.jpg"
#   r = requests.get(imapath)
#   imuria = imgo['imuri']
#   imuria1 = imgo1['imuri']
#   imuria2 = imgo2['imuri']
#   ima = Image.open(StringIO(r.content))
#   Config.imaw, Config.imah = ima.size
#   # Config.imaurl=  '/F5T/oc/Flight-Imagery/work/processed/'+   imuria + "/small.jpg"
#   ts1=tnow()
#   cornersp = np.float32([[0, 0], [Config.imaw, 0], [Config.imaw, Config.imah], [0, Config.imah]])
  
#   cornabp = np.float32(imgo['relabdone']['coords'])
#   if ncluster==3:   cornbcp = np.float32(imgo1['relabdone']['coords'])
 
#   # corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)  )
#   # get a -> b homography in pixels
#   ha1p = cv2.getPerspectiveTransform( cornersp, cornabp)
#   # get c corners in a frame
#   if ncluster==3:   cornacp = np.float32( cv2.perspectiveTransform(cornbcp.reshape(1, -1, 2), ha1p).reshape(-1, 2)  )
#   # get a -> c homography in pixels
#   ha2p= cv2.getPerspectiveTransform( cornersp, cornacp)
#   k2a0, ddesca = getfeat(bina) 

#   print 'ddesca',len(ddesca)
#   if  len(ddesca) < 1:
#     print  'problem ddesca'
#     quit()
#   js0 = getstatus(imuria, stage='georefdone')
#   direct=False
#   print "js0 == 00", js0
#   if js0=='null' or js0 == None :
#     print 'norect'
#     direct=True
#   # imgo['k2a0']=k2a0
#   # imgo['ddesca']=ddesca 
#   # imgo['cornersp']=cornersp 



#   print 'direct'
#   try:
#     qkcenter = oc.lonlat2qk(float(imgo['ocguide']['lon']), float(imgo['ocguide']['lat']), Config.zoom )
#   except:
#     print float(imgo['ocguide']['lon'])
#     quit()
      
#   qk = oc.qk2qk(qkcenter,int(-1),int(-3))  

#   tx,ty, z = oc.qk2xy(qk)
#   minpx , minpy = tx*256 , ty * 256
#   wx, wy = 8, 8
#   k2b, ddescb , Config.ub = mosaicfeatureswh(qk,wx,wy)
#   print 'ddescb',len(ddescb)
#   if  len(ddescb) < 10:
#     print  'problem ddescb'
#     return

#   matches = bf.match(ddesca,ddescb)
#   p1, p2, kp_pairs = filter_matches2(k2a0, k2b, matches, ratio = 0.8)

#   H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 8.0)
#   print 'feats', len(p1)
#   print 'inliers', np.sum(status)
#   imgo['cornersout'] = np.float32( cv2.perspectiveTransform(cornersp.reshape(1, -1, 2), H).reshape(-1, 2)   )
#   print imgo['cornersout']
#   # plotkpinliers(p1,p2,status,x0=0, y0=0 , svgfn='inliersdir.svg', qklist=Config.ub)

#   plotkps(p1, p2, status = [],  svgfn='kpsmf.svg' , scale=0.6, x0=100, y0=100, corners=[cornersp] )
#   # plotkps(p2, p2, status = [],  svgfn='kpswhmf.svg' , scale=1, x0=100, y0=100, corners=[cornersp] )

 
#   return imgo 



def montage(fnames,(nx,ny)):
    # Read in 
    imgs = [Image.open(fn) for fn in fnames]
 
    # Create raw image.
    white = (255,255,255)
    inew = Image.new('RGB',(nx*256,ny*256),white)

    # Insert each thumb:
    for irow in range(ny):
        for icol in range(nx):
            left =  icol*256
            right = left + 256
            upper =   irow*256
            lower = upper + 256
            bbox = (left,upper,right,lower)
            try:
                img = imgs.pop(0)
            except:

                quit()
            inew.paste(img,bbox)
    return inew



def gettilesourceurl(qk, source="ve",ext=".jpg"):
  if (source=='ve'):
    return "http://t0.tiles.virtualearth.net/tiles/a" +qk+".jpeg?g=1398"
  if (source!='ve'):
    return source+"/a" +qk+ext




def qkmontage(qklist, ve='/home/oc/F5T/oc/Flight-Imagery/work/ve/', debug = False):
    txmin, tymin ,tz=oc.qk2xy(qklist[0])
    txmax, tymax ,tz=oc.qk2xy(qklist[-1])

    nx=txmax-txmin+1
    ny= tymax -tymin+1
    # print  nqk, nx, ny
    
    # Create raw image.
    white = (255,255,255)
    inew = Image.new('RGB',(nx*256,ny*256),white)

    # Insert each thumb:
    nt=0
    listqk={}
    prob=0
    if debug:
      print ve
    # quit()
    for qk in qklist:
        # print oc.thuman(),qk
        if ve[:4]=='http':
          imurl = ve+'a'+qk+'.jpg'
          # impng = ve+'a'+qk+'.png'
          img = oc.getimg(imurl)
          # print oc.thuman(),'open',imurl
          # if img==-1:
          #   img = oc.getimg(impng)
          # if img==-1:
          #   continue  
          
        else:
          veroot=ve
          fn=veroot+"a"+qk+".jpg"


          try:
            img=Image.open(fn)
            # print oc.thuman(),'open',qk
          except:
            if debug:
              print oc.thuman(),"curl ve",qk
          #   srcurl = gettilesourceurl(qk, source=tilesource )
          #   system("curl -s "+srcurl+" > "+veroot+"a" + qk+".jpg")
            # system("curl -s http://t0.tiles.virtualearth.net/tiles/a" +qk+".jpeg?g=1398 > "+veroot+"a" + qk+".jpg")
            try:
              system("curl -s http://t0.tiles.virtualearth.net/tiles/a" +qk+".jpeg?g=1398 > "+veroot+"a" + qk+".jpg")
              img=Image.open(fn)
            except:
              continue 
          # print 'qkmontage prob', fn
           
          
            
        # print fn
        nt+=1
        tx, ty ,tz=oc.qk2xy(qk)
        left =  (tx-txmin)*256
        right = left + 256
        upper =   (ty-tymin)*256
        lower = upper + 256
        bbox = (left,upper,right,lower)
        # print qk
        try:
          inew.paste(img,bbox)
        except:
          prob=1
          listqk[qk]=bbox
          if debug:
            print 'qkmontage imgprob', fn  
          
    if prob>0:
      if debug:
        print 'qkmontage', listqk       
    # inew.save('mb.jpg')    
    if debug:
      print "oc.qkmontage", nt, '/', len(qklist)
    return inew
 
def imnum(nd):
    ix=nd%1000;
    iy= int(math.floor(nd/1000))
    if ix ==0 : ix=1
    return ix, iy




















def split(runo, js):


  imfpath = "http://10.0.0.111/F7T/Flight-Imagery/" + js['imuri'] 
  print 'split', imfpath
  # process('curl '+imapath+' > smalltmp.jpg')
  r = requests.get(imfpath, stream=True)
  if r.status_code == 200:
        # r.raw.decode_content = True
        featdeca = r.content
  # ima = Image.open('smalltmp.jpg')
  else:
    print "not found"
    return

 
  try:
    ima = Image.open(StringIO(r.content) )
  except:
    print r.content
    quit()
  Config.impw, Config.imph = ima.size
  # print "imsize", Config.impw, Config.imph 

  
  corners = list2corn( [[0, 0], [Config.impw, 0], [Config.impw, Config.imph], [0, Config.imph]]) 

  ca =  list2corn( js['coords'] )
  
  te=tnow()

  Mapix2ll = cv2.getPerspectiveTransform( corners, ca)

  # print ca
  # print Mapix2ll
 
  # print  ca[0][0] 
  res=(oc.calcdistlonlat((ca[0][0][0],ca[0][0][1]), (ca[0][1][0],ca[0][1][1])))*1000/6016
  # print 'res', res
  # print 'res',(oc.calcdist((ca[0][2][1],ca[0][2][0]), (ca[0][1][1],ca[0][1][0])))*1000/4000

 
  # print 'tileres18', oc.tileres(18)
  # print 'tileres20', oc.tileres(20)
  # print 'tileres22', oc.tileres(22)
  # print 'tileres22', oc.tileres(22)
  z=20
  if res > 0.15:
    z=18

  maxlat=ca[0][2][1]
  minlon=ca[0][2][0]
  minlat=ca[0][2][1]
  maxlon=ca[0][2][0]
 

  for (b,a) in js['coords']:
      # print a,b
      maxlat=max(maxlat, a)
      minlon=min(minlon, b)
      minlat=min(minlat, a)
      maxlon=max(maxlon, b)
  # print " maxlat,minlon ", maxlat,minlon ,  minlat,maxlon 

  tx, ty = oc.LL2Tile(  maxlat,minlon  , z)
  tx2, ty2 = oc.LL2Tile(  minlat,maxlon  , z)
  xb=tx*256
  yb=ty*256
  xb2=(tx2+1)*256
  yb2=(ty2+1)*256

  bw=xb2-xb
  bh= yb2-yb
  # print "bwbh", bw, bh, oc.calcdist([maxlat,minlon],[minlat,maxlon] )
  

  # check if resonable < 10km diag
  if oc.calcdist([maxlat,minlon],[minlat,maxlon] )>10:
      print "too many tiles"
      quit()
  
  img = np.ones((bh,bw,3), np.uint8)
  pts1 = np.float32([[0,0],[6016,0],[6016,4000],[0,4000]])
  pts2 = np.float32([[0,0],[6016,0],[6016,4000],[0,4000]])

  for nc in range(4):
      # print str(nc), js['coords'][nc][0],js['coords'][nc][1]

      px0,py0 = oc.LL2Pixel(js['coords'][nc][1],js['coords'][nc][0], z)
      px1,py1 = oc.LL2Pixel(js['coords'][(nc+1)%4][1],js['coords'][(nc+1)%4][0], z)

      # print "pxy0",px0, py0
      x0=(px0-xb)
      y0=(py0-yb)
      x1=(px1-xb)
      y1=(py1-yb)
      pts2[nc]=[x0,y0]
      # print x0, y0
      
  M = cv2.getPerspectiveTransform(pts1,pts2)

  # print M
  


  # img = cv2.imread( image)
  img = cv2.cvtColor(np.array(ima), cv2.COLOR_RGB2BGR)
  height, width, channels = img.shape
  # print height
  # quit()
  # img = cv2.imread(outname)
  exdata = exifdata(ima)

  apert =exdata['aperture']
  tadj=  exdata['shutter']/4000.0  

  apertadj=(apert**2.25  )/(5**2.25   )
  print apertadj, apertadj*tadj
  img1 = dcombo(img, ratio= apertadj*tadj)
  # img1 = dcombo(img, ratio=(6**2 /apert**2 )  )
  img1c = clahergb(img1,cr=1.4,cg=1.4,cb=1.4)

 

  dst=cv2.warpPerspective(img1c,M,(bw,bh))
  # ret,thresh1 = cv2.threshold(dst[:,:,1],1,255,cv2.THRESH_BINARY)
  # png = np.zeros((bh,bw,4), np.uint8)
  # png[:,:, 0 ] =     dst[:,:, 0 ] 
  # png[:,:, 1 ] =     dst[:,:, 1 ]     
  # png[:,:, 2 ] =     dst[:,:, 2 ] 
  # png[:,:,3 ]=thresh1


  imstub = js['imuri'].replace('/','')
  res = system('mkdir -p '+runo['outdir'] + imstub)

  existnames = os.listdir(runo['outdir'] + imstub)
 
  for name in existnames:
    if name[:3] == 'a02':
      print 'name',name
      system('rm '+runo['outdir'] + imstub+'/a023* ')
      break

    
 
   
  

  cv2.imwrite(runo['outdir']+imstub+'/apng.jpg', dst)
  # cv2.imwrite('apng.png', dst)

  hdst, wdst, ddst = dst.shape


  # print "outpath", outpath
  # print "====", impath, "js ", js['impath'], js['coords'][0][1], js['coords'][0][0]

  nti=0
  # print Config.tiles+imstub+"/tilelist.txt"
  
  with open(runo['outdir']+imstub+"/tilelist.txt", "w") as myfile:
              myfile.write( "")  
  with open(runo['outdir']+imstub+"/tilejson.txt", "w") as myfile:
              myfile.write(json.dumps(js))

  deltile = runo['outdir']+imstub+"/a02*.jpg"   
  junk =    runo['outdir']+imstub+"/junk/"
  # system('mkdir -p  '+junk)
 
  # system('mv '+deltile+' '+junk)
           
  for y in xrange(hdst/256):
 
    for x in xrange(wdst/256):
 
      qk=oc.TXY2Quadkey(tx+x,ty+y,z)
      px=x*256
      py=y*256
      # print "xy", x,y, px,py
      

      crop_img = dst[py:py+256, px:px+256] # Crop from x, y, w, h -> 100, 200, 300, 400
      
      if not  oc.point_inside_polygon( px,py, pts2): continue
      if not  oc.point_inside_polygon( px+256,py, pts2): continue
      if not  oc.point_inside_polygon( px+256,py+256, pts2): continue
      if not  oc.point_inside_polygon( px,py+256, pts2): continue

      # print pts2
      # oc.svgqk(qk)
      
      tilepathname = runo['outdir']+imstub+"/a"+qk+".jpg"
      cv2.imwrite(tilepathname, crop_img)
      st =os.stat(tilepathname)
      sz=st.st_size
      if sz< 5000:
        print sz, basein+name

      with open(runo['outdir']+imstub+"/tilelist.txt", "a") as myfile:
          myfile.write( "a"+qk+".jpg"+"\n")


      # pil_im = Image.fromarray(crop_img)    
      # kpn, feats=findfeatures(pil_im)


      # featpathname = tilepathname+'.bin'

      # with open(featpathname, 'wb') as f:
      #   f.write(feats)



      nti+=1    
 

  print "save",nti,"tiles" 
  return runo['outdir']+imstub+"/"










def mix(indir, outmix):
 
  names = os.listdir(indir)
 
  print "nfiles", len(names)

  ncopy=0
  for f in sorted(names):
    if f[-3:] != 'jpg' or f[:3] != 'a02' :
      continue
    qkreq = f[1:-4] 
    src=indir+'/a'+qkreq+'.jpg'
    dst=outmix+'/a'+qkreq+'.jpg'
    info = outmix+'/a'+qkreq+'.jpg.txt'
    junk =  outmix+'/junk/'
    # system('mkdir -p  '+junk)
    # if os.path.isfile(outmix+'/a'+qkreq[:10]+'.png'):
    #   deltile = outmix+'/a'+qkreq[:10]+'*.png'
    #   # system('mv '+deltile+' '+junk)
 
 
    copy2(src,dst)

    ta= arrow.utcnow()
 
 
   

    lont, latt = oc.qk2lonlat(qkreq)
    with open(info, mode='w') as file:
      file.write('{ "ts":'+"%0.6f"%tnow()+', "src": "'+src+'" , "lon":'+str(lont)+',"lat":'+str(latt)+', "thuman": "' +thuman()+  '"}' ) 
    # print 'wrote', info  
    ncopy+=1  

  print 'ncopy' , ncopy 
  return    








def unmix(indir, outmix):
 
  names = os.listdir(indir)
 
  print "nfiles", len(names)

  ncopy=0
  for f in sorted(names):
    if f[-3:] != 'jpg' or f[:3] != 'a02' :
      continue
    qkreq = f[1:-4] 
    src=indir+'/a'+qkreq+'.jpg'
    dst=outmix+'/a'+qkreq+'.jpg'
    deldst=outmix+'/dela'+qkreq+'.jpg' 
    info = outmix+'/a'+qkreq+'.jpg.txt'
    delinfo = outmix+'/dela'+qkreq+'.jpg.txt'
    print 'unmix', dst 
    try:
        ss = os.stat(dst)
        print ss.st_size
        shutil.move(dst,deldst)
        shutil.move(info,delinfo)
      #       with open(info, mode='w') as file:
      # file.write('{ "ts":'+"%0.6f"%tnow()+', "src": "'+src+'" , "lon":'+str(lont)+',"lat":'+str(latt)+', "thuman": "' +thuman()+  '"}' ) 
    except:
      pass   
    
    ta= arrow.utcnow()
    lont, latt = oc.qk2lonlat(qkreq)
  
    ncopy+=1  

  print 'ndel' , ncopy 
  return    





def dcombo(image, gamma=2.25, ratio=1.0):
  inv  = 1.0 / gamma
  lutf = np.array([min(1.0, ((  ((i / 255.0) ** gamma) * ratio)** inv ))*255 
    for i in np.arange(0, 256)])

  lut=lutf.astype("uint8")
  l31 = lut[100]

  for n in range(100):
    lut[n] = l31*n/100
  lut[255]=255
  lut[254]=255
  lut[253]=255
  lut[252]=255

  lut[251]=255
  lut[250]=255

  # .astype("uint8") 
  return cv2.LUT(image, lut)


def exifdata(img):
  exif = img._getexif()
  DateTimeOriginal="1900"
  FocalLength=(10,10)
  ExposureTime=(10,10)
  FNumber=(10,10)
  ISOSpeedRatings=0

  for tag, value in exif.items():
      decoded = TAGS.get(tag, tag)
      # if decoded!="MakerNote": print "decoded", decoded, value
      if decoded=="DateTimeOriginal": DateTimeOriginal=value
      if decoded=="FocalLength": FocalLength=value
      if decoded=="ExposureTime": ExposureTime=value
      if decoded=="FNumber": FNumber=value
      if decoded=="ISOSpeedRatings": ISOSpeedRatings=value
      # FocalLength
      # ExposureTime
      # MaxApertureValue
      # ret[decoded] = value
  # print  dirName+"/"+fname, float(FNumber[0])/float(FNumber[1]),(ExposureTime[1])/(ExposureTime[0])

  # exdata= imuri+"," +DateTimeOriginal+"," +str(FocalLength[0]/FocalLength[1])+"," +str(float(FNumber[0])/float(FNumber[1])) + ","+str((ExposureTime[1])/(ExposureTime[0]))+","+str(ISOSpeedRatings)
  # exdata=   str(FocalLength[0]/FocalLength[1])+"," +str(float(FNumber[0])/float(FNumber[1]))  + ","+str((ExposureTime[1])/(ExposureTime[0]))

  exdata = {  'date' : DateTimeOriginal, 'focal' : (FocalLength[0]/FocalLength[1]),'aperture' : (float(FNumber[0])/float(FNumber[1])) , 'shutter' : ((ExposureTime[1])/(ExposureTime[0])), 'iso': (ISOSpeedRatings)  }
  return  exdata
 











def getimgsize(imuri, processed = "http://10.0.0.4/F5T/oc/Flight-Imagery/work/processed/"):
  imapath = processed + imuri  + "/small.jpg"

  r = requests.get(imapath)

  try:
    ima = Image.open(StringIO(r.content))
    Config.impw, Config.imph = ima.size
  except:
    print "getimgsize not found", imapath
    Config.impw, Config.imph = 6016*0.16, 4000*0.16
 
 
  
  return np.float32([[0, 0], [Config.impw, 0], [Config.impw, Config.imph], [0, Config.imph]])








def getrelab(imgoa , imgob, debug=False):

  # k2a, ddesca = oc.getfeat('http://10.0.0.4/F5T/oc/Flight-Imagery/work/processed'+ imgoa ['imuri']+'/feats.bin')
  # k2b, ddescb = oc.getfeat('http://10.0.0.4/F5T/oc/Flight-Imagery/work/processed'+ imgob ['imuri']+'/feats.bin')
  # kpa, ddesca = imgoa['kpa'], imgoa['ddesc']
  # kpb, ddescb = imgob['kpa'], imgob['ddesc']

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  print "getrelab", imgoa['imuri'] , imgob['imuri'] , len(imgoa['ddesc']), len(imgob['ddesc'])
  if len(imgoa['ddesc']) < 80 or len(imgob['ddesc'])<80:
    # print 'getrelab a & b', len(imgoa['ddesc']), len(imgob['ddesc'])
    return "pddesc"
  matches = bf.match(imgoa['ddesc'],imgob['ddesc'])
  # p1, p2, kp_pairs = filter_matches2(kptf, k2b, matches, ratio = 0.8)
  p1, p2, kp_pairs = filter_matches2(imgoa['kp'], imgob['kp'], matches)
  H, status = cv2.findHomography(p2, p1, cv2.RANSAC, 8.0)
  # Hr = cv2.estimateRigidTransform(p2, p1,False)
  print "imgoa['imuri'] ",imgoa['imuri']
  corners = getimgsize(imgoa['imuri'] )
 
  crel = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)  
 
  # crig=    cv2.transform(corners.reshape(1, -1, 2), Hr).reshape(-1, 2)  
  mo={ 'p1':p1, 'p2':p2 , 'corners': [corners, crel]}

  imout=  {"imuria":  imgoa ['imuri'],"imurib": imgob ['imuri'],'coords': crel.tolist(), 'orig': corners.tolist() , 'inliers': np.sum(status)}


  if debug:
    print 'relab', imgoa['imuri'], imgob['imuri']
    print 'len(k2a)' ,len(k2a) ,len(k2b)  
    print 'inliers' , np.sum(status)
    print H
    oc.plotpairs(mo,  svgfn='kpmatchrelab.svg')

  return imout






def getcenter(imuri):
    r =  requests.get("http://10.0.0.4/repo/getstatus.php?stage=ocguide&imuri=" +imuri)
    js =json.loads(r.text )
    print 'getcenter', float(js['lon']), float(js['lat'])
    return float(js['lon']), float(js['lat'])



def getoutdir(imuri):
    rr = imuri.split('/')
    outdir = '/home/oc/F5T/oc/Flight-Imagery/work/tiles/'+rr[1]+'/'
    imstub =  imuri.replace('/','')
    res = system('mkdir -p '+outdir + imstub) 
    return outdir + imstub  +'/' 





def resizetiles(qk, base,  inext=".jpg",inprefix='a', timg = "aa.png", debug=False):
  "Read resize and save"

  imf= base+qk+inext
  if debug: print("make " +imf)
  
  scale=len(qk)
  if debug:  print ("scale:" +str(scale))
 
  im0f=base+inprefix+qk+"0"+inext
  im1f=base+inprefix+qk+"1"+inext
  im2f=base+inprefix+qk+"2"+inext
  im3f=base+inprefix+qk+"3"+inext


  im=Image.open(timg)
 
  n=0
  i0=1
  i1=1
  i2=1
  i3=1


  try:
    im0 = Image.open(im0f)
    n+=1   
  except:
    i0=0

  try:  
    im1 = Image.open(im1f)
    n+=1
  except:
    i1  = 0

  try:  
    im2 = Image.open(im2f)
    n+=1
  except:
    i2  = 0

  try:
    im3 = Image.open(im3f)
    n+=1
  except:
    i3  = 0

  # print "good images", n
  box0 = (0, 0, 128, 128)
  box1 = ( 128, 0,256,128)
  box2 = (0, 128, 128, 256)
  box3 = (128, 128,256, 256)
  region0 = im.crop(box0)
  region1 = im.crop(box1)
  region2 = im.crop(box2)
  region3 = im.crop(box3)

  if i0 is 1: 
    out0 = im0.resize((128, 128), Image.BICUBIC)
    im.paste(out0, box0)
  if i1 is 1: 
    out1 = im1.resize((128, 128), Image.BICUBIC)
    im.paste(out1, box1)
  if i2 is 1: 
    out2 = im2.resize((128, 128), Image.BICUBIC)
    im.paste(out2, box2)
  if i3 is 1: 
    out3 = im3.resize((128, 128), Image.BICUBIC)
    im.paste(out3, box3)

  return im, n



def smartshrink(basein,baseout,zoomin=18, newer=True):
  n=0
  countmod=0

      
  makelist={}
  tot = 0
  tz20 = 0
  tz18 = 0

  for zz in xrange(zoomin,10, -1):
    print "zz", zz
    names = sorted(os.listdir(baseout))
    print  baseout, len(names)
    for fn in names:


      if fn[-3:]!='jpg' and fn[-3:]!='png' :
        # print 'not image', fn
        continue
      
      if zz!=len(fn[1:-4]):
        continue
      # print 'good', zz, len(fn[1:-4]), fn
      print fn
   
      n+=1
   
      qk = fn[1:-4]
      qkmake = fn[1:-5]

      countmod+=1

      qkmakename="a"+qkmake+".png"
      try:
        tileinfofile=baseout+fn+'.txt'

        with  open(tileinfofile, "r") as mf:
          lines = mf.readlines()


        jstile=json.loads(lines[0])
        tstile=jstile['ts']
      except:
        tstile=os.stat(baseout+fn).st_mtime
        # with open(tileinfofile, mode='w') as f:
        #  f.write('{"ts":'+"%0.6f"%tstile +'}' ) 
   
      #   print 'noinfo for', tileinfofile  
      try:
        makeinfofile=baseout+qkmakename+'.txt'
        with  open(makeinfofile, "r") as mf:
          lines = mf.readlines()
        jsmake=json.loads(lines[0])
        tsmake=jsmake['ts']
        # print tileinfofile, "%0.6f"%tstile
        # print makeinfofile, "%0.6f"%tsmake
        if tsmake>tstile  :
          # print 'make is newer'
          continue
      except:
        pass


      shrinktime=0
      newtime=0
   

      if qkmakename in makelist:
        print 'already made'
        continue

      #    print 'making', zz, qkmake
      inext=".png"
      if zz==zoomin:
        inext='.jpg'
   
   
      im, nt = oc.resizetiles(qkmake, basein,   inext=inext, timg = "aa.png")
       
      makelist[qkmakename]=1
      if n%100 is 1:
        print n, qkmake 


      im.save(baseout+qkmakename, "PNG")
      info = baseout+'/'+qkmakename+'.txt'
      # print info
   
      with open(info, mode='w') as f:
         f.write('{"ts":'+"%0.6f"%tnow()+'}' ) 


def obj2list(obj):

  l2 = []
  for ll in sorted(obj):
    l2.append(ll)
  return l2  





def qklistshrink(qklistfn,basein,baseout,zoomin=18, newer=True, ext='.jpg',minzoom=10):
  n=0
 
  countmod=0
  ts=tnow()
 
       
  makelist={}
  tot = 0
  tz20 = 0
  tz18 = 0
 
 
  with open(qklistfn, mode='r') as f:
    lines=f.readlines()
 
  worklist = {}  
  for  line in lines:
    qk=line.strip()
    worklist[qk]=(qk)
 
  qkl = obj2list(worklist)  
 
  for zz in xrange(zoomin,minzoom, -1):
 
    qklnext={}
    for  qk in sorted(qkl):
      n+=1
      qkmake = qk[:-1]
      qklnext[qkmake]= qkmake
      outmsg = str(zz) + ' ' + qk
      qkmakename="a"+qkmake+".jpg"
      # try:
      #   tileinfofile=baseout+qkmakename+'.txt'
      #   with  open(tileinfofile, "r") as mf:
      #     lines = mf.readlines()
 
      #   jstile=json.loads(lines[0])
      #   tstile=jstile['ts']
      # except:
      #   tstile=os.stat(baseout+fn).st_mtime
 
      # try:
      #   makeinfofile=baseout+qkmakename+'.txt'
      #   with  open(makeinfofile, "r") as mf:
      #     lines = mf.readlines()
      #   jsmake=json.loads(lines[0])
      #   tsmake=jsmake['ts']
      #   # print tileinfofile, "%0.6f"%tstile
      #   # print makeinfofile, "%0.6f"%tsmake
      #   if tsmake>tstile  :
      #     print outmsg ,  'make is newer'
      #     continue
      # except:
      #   pass
 
 
      shrinktime=0
      newtime=0
    
 
      if qkmakename in makelist:
        # print outmsg[:-1] ,'already made'
        continue
 
      #    print 'making', zz, qkmake
      inext=".jpg"
      if zz==zoomin:
        inext='.jpg'
    
    
      im, nt = oc.resizetiles(qkmake, basein,   inext=inext, timg = "aa.png")
        
      makelist[qkmakename]=1
      if n%10 is 1:
        print qkmakename, len(makelist),n 
      im.save(baseout+qkmakename, "JPEG")
      info = baseout+'/'+qkmakename+'.txt'
      # print info
    
      # with open(info, mode='w') as f:
      #    f.write('{"ts":'+"%0.6f"%tnow()+'}' )   
 
    qkl=obj2list(qklnext)  
 
  te=tnow()  
  if len (makelist)>0:
    print 'qklistshrink made'  , len(makelist), (te-ts), (te-ts)/len(makelist)
  else:
    coutr('qklistshrink made nothing') 

    
  return 
 
 
 
 























def qklistshrinkn(qklistfn,basein,baseout,zoomin=18, newer=True, ext='.jpg', outext='.png', outprefix='a',  minzoom = 10):
  n=0

  countmod=0
  ts=tnow()
  if outext=='.jpg':
    outtype='JPEG'
  if outext=='.png':
    outtype='PNG'
  makelist={}
  tot = 0
  tz20 = 0
  tz18 = 0


  with open(qklistfn, mode='r') as f:
    lines=f.readlines()
  print 'qklistshrink len ',len(lines)

  worklist = {}  
  for  line in lines:
    qk=line.strip()
    worklist[qk]=(qk)

  qkl = obj2list(worklist)  

  for zz in xrange(zoomin,minzoom, -1):

    qklnext={}
    inprefix='a'
    for  qk in sorted(qkl):
      n+=1
      qkmake = qk[:-1]
      qklnext[qkmake]= qkmake
      outmsg = str(zz) + ' ' + qk
      qkmakename=outprefix+qkmake+outext
      # try:
      #   tileinfofile=baseout+qkmakename+'.txt'
      #   with  open(tileinfofile, "r") as mf:
      #     lines = mf.readlines()

      #   jstile=json.loads(lines[0])
      #   tstile=jstile['ts']
      # except:
      #   tstile=os.stat(baseout+fn).st_mtime

      # try:
      #   makeinfofile=baseout+qkmakename+'.txt'
      #   with  open(makeinfofile, "r") as mf:
      #     lines = mf.readlines()
      #   jsmake=json.loads(lines[0])
      #   tsmake=jsmake['ts']
      #   # print tileinfofile, "%0.6f"%tstile
      #   # print makeinfofile, "%0.6f"%tsmake
      #   if tsmake>tstile  :
      #     print outmsg ,  'make is newer'
      #     continue
      # except:
      #   pass


      shrinktime=0
      newtime=0
   

      if qkmakename in makelist:
        # print outmsg[:-1] ,'already made'
        continue

      #    print 'making', zz, qkmake
      # inext=".png"
      # if zz==zoomin:
      #   inext='.jpg'
   
   
      im, nt = oc.resizetiles(qkmake, basein,   inprefix=inprefix, inext=jpg, timg = "aa.png")

 
       
      makelist[qkmakename]=1
      if n%10 is 1:
        print baseout+qkmakename, len(makelist),n 
      if nt < 1:
        print "notsaved",nt, qkmakename
        continue  

      print " saved",nt, baseout+qkmakename  
      im.save(baseout+qkmakename, outtype)
      info = baseout+'/'+qkmakename+'.txt'
      # print info
   
      with open(info, mode='w') as f:
         f.write('{"ts":'+"%0.6f"%tnow()+'}' )   

    qkl=obj2list(qklnext) 
    inprefix='z' 

  te=tnow()  
  print 'made'  , len(makelist), (te-ts)
  print (te-ts)/len(makelist)
  return  










   
def jsonencode(obje):
    return str(obje).replace("u'", " '").replace("'",'"')






def dncDSC2DB(imuri, fi,processed  ,AKAZEThreshold=0.0005, maxn=0, testdnc=0, scale=0.6):
  """
  - resize image
  - detect and compute AKAZE features
  - write bin
  """
 
  inpath = fi + imuri
  processed_path =  processed + imuri +'/'
 
 
  res = process('mkdir -p '+ processed_path)
  
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)

  try: 
    r = requests.get(inpath)
    im = Image.open(StringIO(r.content))
    (width, height) = im.size
    swidth, sheight = width*scale,height*scale
    im.thumbnail( (swidth, sheight ), Image.ANTIALIAS)
  except:  
    print 'imge dl prob', inpath
    quit()
 

  img1 = np.array(im) 
  img1 = img1[:, :, ::-1].copy() 
  # img1 = cv2.imread(inpath)
  # small = cv2.resize(img1, (0,0), fx=0.16, fy=0.16) 
  
  cv2.imwrite(processed_path+"small.jpg", img1)
  cv2.imwrite("small.jpg", img1)
  ts=tnow() 
  kp1, desc1 = detector.detectAndCompute(img1, None)

  te=tnow()
  timetaken = te-ts
  print len(kp1), "kp", te-ts , "s"
  # ['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'angle', 'class_id', 'octave', 'pt', 'response', 'size']
  feats=''

  n1=0
  for nkp in range(len(kp1)):
    kp = kp1[nkp]
    float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
    keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
 
    g=keyp+desc1[nkp].tostring()
    ss=np.fromstring(g[-61:], dtype=np.uint8)
    feats += g
    n1+=1

  with open(processed_path+'feats.bin', 'wb') as f:
    f.write(feats) 
  r = requests.get("http://10.0.0.4/repo/dncdsc.php?stage=dncDSC2DBdone&imuri="+imuri+"&kp="+str(len(kp1))+"&timetaken="+str(timetaken)+"&width="+str(swidth)+"&height="+str(sheight)+"&status=good&AKAZEThreshold=0.001" )








def encodeakazefeatures(kpa, desc):
    feats=''
 
    for nkp in range(len(kpa)):
        kp = kpa[nkp]
        float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
        keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
 
        feats += keyp+desc[nkp].tostring()
 
    return feats


def resize(img1, scale):
    if scale != 1.0:
        dim = (   int(img1.shape[1] * scale), int(img1.shape[0] * scale) ) 
        img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    return img1     

 
 
 

def dnc(img1 , scale=1.0, AKAZEThreshold=0.0005  ):
  """
  - resize image
  - detect and compute AKAZE features
  - store in db in px coords
  - update DSC DNC flag in db
  """
  if scale != 1.0:
    img1 = resize(img1, scale)
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)
  kp, desc = detector.detectAndCompute(img1, None)
  # print len(kp)
  if len(kp) < 1:
    return {'kp':[], 'desc': [] , 'shape' : img1.shape}

 
  return {'kp':kp, 'desc': desc , 'shape' : img1.shape}
 

def dncimuri(imuri , scale=0.16, AKAZEThreshold=0.0005  ):
  """
  - resize image
  - detect and compute AKAZE features
  - store in db in px coords
  - update DSC DNC flag in db
  """
  print "dncimuri", getimuriloc(imuri)
  loc = oc.getimuriloc(imuri)
  r = requests.get(loc)
  imgpil = Image.open(StringIO(r.content))
  img1 = np.array(imgpil)

  # img1=cv2.imread(imuri)
  if scale != 1.0:
    img1 = resize(img1, scale)
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)
  kp, desc = detector.detectAndCompute(img1, None)
  # print len(kp)
  if len(kp) < 1:
    return {'kp':[], 'desc': [] , 'shape' : img1.shape}

 
  return {'kp':kp, 'desc': desc , 'shape' : img1.shape}

 

def match(desc1,desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf.match(desc1,desc2)























def RANSACUpdateNumIters(p, ep, maxIters):
    p, ep = min(max(p,0.),1.), min(max(ep,0.),1.)
    num = max(1.-p, sys.float_info.min)
    denom = 1.-pow(1.-ep, MODELPOINTS)
    if denom < sys.float_info.min: return 0
    num , denom = np.log(num), np.log(denom)
    return maxIters if denom>=0 or -num>=maxIters*(-denom) else round(num/denom)



def computeError(m1,m2,model,err):
    # global CETIME
    m1Projected = cv2.perspectiveTransform(np.array([np.array(m1, dtype='float32')]), np.array(model, dtype='float32'))[0]
    # asdf = time.clock()
    err = np.sqrt(np.sum(ne.evaluate("(m1Projected-m2)**2"), axis=1).tolist())
    # CETIME = CETIME + time.clock()-asdf
    return err




def findInliers(m1,m2,model,err,thresh,mask):
    err = computeError( m1 , m2 , model, err)
    # t2 = thresh*thresh
    t2 = thresh
    mask = [1 if val < t2 else 0 for val in err]
    return mask
 
# class color:
#    PURPLE, CYAN, DARKCYAN, BLUE, GREEN, YELLOW, RED, BOLD, UNDERLINE, END = \
#     '\033[95m', '\033[96m', '\033[36m', '\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[1m', '\033[4m', '\033[0m'


def findshift(m1, m2, thresh, twoSectionMode=False, photoCorners=[], finalLMEDS=False, rad1=40, rad2=10):
    result, niters, maxGoodCount  = False, 222, 0
    origm1, origm2 = np.copy(m1), np.copy(m2)
    # m1,m2 = filterOutCenterPoints(m1,m2)
    mask = np.ones(len(m1),dtype=np.int)
    count = len(m1)
 
    bestMask0 = bestMask = mask
    err = np.zeros(len(m1), dtype='float32')
    bestErr = err
 
    tryCount = 0
    bestdx=-200
    bestdy=-200
    goodtries = 0
 
    while maxGoodCount < 100 and tryCount < 10000:
        dx=random.randint(-rad1,rad1)
        dy=random.randint(-rad1,rad1)
        model = np.float32([[1,0,dx],  [0,1,dy],[0,0,1]])
        mask = findInliers( m1, m2, model, err, thresh, mask )
 
        goodCount = sum(mask)
 
        if goodCount >  maxGoodCount :
            goodtries+=1
            sw = np.copy(mask)
            mask = np.copy(bestMask)
            bestMask = np.copy(sw)
            bestErr = np.copy(err)
            bestModel = np.copy(model)
            maxGoodCount = goodCount
            bestdx=dx
            bestdy=dy
 
 
        tryCount = tryCount+1
    # print color.BOLD, color.YELLOW,"final tryCount",tryCount,color.END,color.END
 
 
    # print "inliers:",color.BOLD+color.YELLOW,sum(bestMask),color.END+color.END, 
    # print "goodtries",color.YELLOW,goodtries,color.RED, bestdx, bestdy, color.END
    # print model

    model, bestMask0, bestdx, bestdy  =  findshiftrefine(m1, m2 ,thresh,  bestdx, bestdy)
    return model, bestMask0, bestdx, bestdy 




def findshiftrefine(m1, m2, thresh, dx0, dy0, radius=8):
    result, maxGoodCount  = False,   0
    origm1, origm2 = np.copy(m1), np.copy(m2)
    # m1,m2 = filterOutCenterPoints(m1,m2)
    mask = np.ones(len(m1),dtype=np.int)
    count = len(m1)
    bestMask0 = bestMask = mask
    err = np.zeros(len(m1), dtype='float32')
    bestErr = err
 
    tryCount = 0
    bestdx=-200
    bestdy=-200
    goodtries = 0
    radius = 10
    for dy in range(-radius,radius):
      for dx in range(-radius,radius):
 
        model = np.float32([[1,0,dx+dx0],  [0,1,dy+dy0],[0,0,1]])
        mask = findInliers( m1, m2, model, err, thresh, mask )
 
        goodCount = sum(mask)
 
        if goodCount >  maxGoodCount :
            goodtries+=1
            sw = np.copy(mask)
            mask = np.copy(bestMask)
            bestMask = np.copy(sw)
            bestErr = np.copy(err)
            bestModel = np.copy(model)
            maxGoodCount = goodCount
            bestdx=dx+dx0
            bestdy=dy+dy0
 
        tryCount = tryCount+1
    # print color.BOLD, color.YELLOW,"final tryCount",tryCount,color.END,color.END
    if maxGoodCount > 0:
        if np.all(bestMask) != np.all(bestMask0):
            if len(bestMask) == len(bestMask0):
                bestMask0 = np.copy(bestMask)
            else:
                bestMask0 = cv2.transpose(bestMask)
            model = np.copy(bestModel)
 
    else:
        model = []
 
    # print "inliers:",color.BOLD+color.YELLOW,sum(bestMask0),color.END+color.END, 
    # print "goodtries",color.YELLOW,goodtries,color.RED, bestdx, bestdy, color.END
    # print model
    return model, bestMask0, bestdx, bestdy 






def makeseriescluster(series ):
  imgoa = getseriesfeats(series)
  imgoa = getseriesrelab(imgoa)
  centers=[]
  corners=[]
  imlist=[]
  for im in imgoa:
    imlist.append(imgoa[im]['imuri'])
  imlist= sorted(imlist)

  kp= imgoa[imlist[0]]['kp']
  ddesc=  imgoa[imlist[0]]['ddesc']
  try:
    corners = [np.float32(imgoa[imlist[0]]['relab']['orig'])]
  except:
    pass
  Hcumu = np.eye(3, dtype=float)
  
  for nc in range(len(imlist)-1):
    # print 'makeseriescluster', imlist[nc], imlist[nc+1]
    # imgolist[imuri] = getimo(imuri)
    cornersp = np.float32(imgoa[imlist[nc]]['relab']['orig'])
    cornabp = np.float32(imgoa[imlist[nc]]['relab']['coords'])
    Hinc = cv2.getPerspectiveTransform( cornersp, cornabp)
    Hcumu = np.dot(Hcumu,Hinc)
    kp.extend(xformkp(imgoa[imlist[nc+1]]['kp'], Hcumu))
    # centers.append((float(imgolist[imuri]['ocguide']['lon']), float(imgolist[imuri]['ocguide']['lat'])))
    c2 = np.float32(cv2.perspectiveTransform(cornersp.reshape(1, -1, 2), Hcumu).reshape(-1, 2) )
    corners.append(c2)
    
    ddesc = np.append(ddesc, imgoa[imlist[nc+1]]['ddesc'], axis=0)
 
 
  print 'makeseriesclusterk2a',  len(kp),  len(ddesc)  
  # plotkp([k2a], svgfn='k2a.svg', x0=800, y0=1600, poly = [])    
  return imlist, kp, ddesc, centers, corners    
 

def getseries(base = '/20151121/mf00/', imb=50, n=2, step=1):
  im=[]
  for nn in range(n):
    nd = imb + nn*step
    ix,iy = oc.imnum(nd)
 
    im.append( base + '1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG')
  return im  


def getseries2(base = '/20151121/mf00/', imb=50, nf=2, nb=2, step=1):
  im=[]
  for nn in range(n):
    nd = imb + nn*step
    ix,iy = oc.imnum(nd)
 
    im.append( base + '1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG')
  return im  


def getmfs(date = '20151121', imb=50 ):
  delays=[0,0,0,0,0]
  if date=="20150124":
    delays = [0,85,67,67,85]
  mfl =['wf00','mf00','mf01','mf02','mf03'  ]
  im=[]
  ix,iy = oc.imnum(imb)
  for i in range(5):
  # for mf in mfl:
    mf= mfl[i]
    ix,iy = oc.imnum(imb+delays[i])
  
    im.append( '/'+date +'/'+ mf+ '/1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG')
  return im  


def getseriesrelab(imgoa):
  imlist=[]
  for im in imgoa:
    imlist.append(imgoa[im]['imuri'])
  imlist= sorted(imlist)
  for nn in range(len(imlist)-1):
    ima = imlist[nn]
    imb = imlist[nn+1]
    # print 'ima,imb',ima,imb
    relab = oc.getrelab(imgoa[ima],imgoa[imb])
    imgoa[imlist[nn]]['relab']= relab
  return imgoa


def getseriesfeats(series):
  imgoa ={}
  for im in range(len(series)):
    imgo= {'imuri': series[im]}
 
    imgo['kp'], imgo['ddesc'] =  oc.getfeat('http://10.0.0.4/F5T/oc/Flight-Imagery/work/processed'+ series[im]+'/feats.bin')
    print 'getseriesfeats', series[im], len(imgo['kp']), len(imgo['ddesc'] )
    imgoa[series[im]] = imgo
  return imgoa  

def getseriesqklist(imgoa):
  imlist=[]
  for im in imgoa:
    imlist.append(imgoa[im]['imuri'])
  imlist= sorted(imlist)
 
  # r = requests.get("http://10.0.0.4/repo/getimuri.php?imuri="+imgoa[imlist[0]]['imuri'])
  # print r.url 
  # r = requests.get("http://10.0.0.4/repo/getstatus.php?stage=georefdone&imuri="+imgoa[imlist[0]]['imuri'])
  # print r.url,r.text 
  # if r.text == 'null':
  #   print 
  # quit()
 



def plotkpinlierslonlat(kp1list=[] ,status=[], x0=0, y0=0,svgfn='kpstatus.svg', qklist=['0230102102223200'], corners= [], imuri='', info="plotkpinliersv2"):
  print 'plotkpinlierslonlat' 
  # print len(kp1list[0]),len(kp1list[1]),len(status)
 
  xo,yo, z = qk2pxpy(qklist[0])

  svgout = ''
  qklist0=[]

  if qklist[0][0]=='a':
    for qkstr in qklist:
      qklist0.append(qkstr[1:-4])
    qklist = qklist0  


  tx0, ty0, z=qk2xy(qklist[0] )
  x0=tx0*256
  y0=ty0*256
 
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      # dxa=tx-tx0
      # dya=ty-ty0
      dxa=tx 
      dya=ty 

      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str(dxa*256-x0)+'px" y="'+str(dya*256-y0)+'px" height="256px" width="256px" style=" opacity:0.9;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256+dx)+'px" y="'+str(dya*256+dy)+'px" height="256px" width="256px" style=" opacity:1;"/>';
      # svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/mb/a'+qk+'.png" x="'+str(dxa*256)+'px" y="'+str(dya*256)+'px" height="256px" width="256px" style=" opacity:1;"/>';


  # with open(svgfn, mode='w') as file:
  #     file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate('+str(-xo)+','+str(-yo)+')">')   
  with open(svgfn, mode='w') as file:
      file.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform="translate(0,0)">') 
 
  with open(svgfn, mode='a') as ff:
        ff.write(svgout )







  linetype=['stroke:rgba(255,0,0,0.8);stroke-width:2;',
    'stroke:rgba(220,0,220,0.6);stroke-width:2;',
    'stroke:rgba(00,0,220,0.6);stroke-width:2;',
    'stroke:rgba(220,220,0,0.6);stroke-width:2;',
    'stroke:rgba(225,210,210,0.6); stroke-width:2;',
    'stroke:rgba(0,0,0,0.8); stroke-width:2;'
    ] 
  nc=0    
  for corn in corners:
    print 'plotkpinlierslonlat corn'
    print  corn

    
    for cc in range(len(corn)):
      cnext = corn[(cc+1)%len(corn)]
      cnow = corn[cc]
      x1 , y1 = cnow[0]-x0, cnow[1]-y0
      x2 , y2 = cnext[0]-x0, cnext[1]-y0
      s3='stroke:rgba(255,220,0,0.7); stroke-width:'+str(nc+1)+';'
      s3=linetype[nc%len(linetype)]
      aa=''
      print 'plotkpinlierslonlat cnow', cnow, 'cnext', cnext
      x1g, y1g = lonlat2pxpy(cnow[0], cnow[1], z)
      x2g, y2g = lonlat2pxpy(cnext[0], cnext[1], z)

      # aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+s3+'"  />'
      aa+='<line x1="'+str(x1g-x0)+'" y1="'+str(y1g-y0)+'" x2="'+str(x2g-x0)+'" y2="'+str(y2g-y0)+'" style="'+s3+'"  />'
      with open(svgfn, mode='a') as ff:
        ff.write(aa )
    nc+=1
 









  style=['stroke:rgba(255,222,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;',
    'stroke:rgba(200,0,200,0.8); fill:rgba(200,0,255,0.2);stroke-width:1;',
    'stroke:rgba(0,250,0,0.6); fill:rgba(55,55,220,0.2);stroke-width:2;',
    'stroke:rgba(255,255,0,0.6); fill:rgba(255,255,0,0.6);stroke-width:1;',
    'stroke:rgba(220,0,0,0.6); fill:rgba(155,0,250,0.6);stroke-width:1;',
    'stroke:rgba(225,210,210,0.6); fill:rgba(255,0,150,0.2);stroke-width:1;',
    'stroke:rgba(0,0,0,0.8); fill:rgba(255,255,0,0.2);stroke-width:1;'
    ]
  ns=0  

  nbad=0
  ngood=0
  
 
  try:
    p1=kp2p(kp1list[0])
    p2=kp2p(kp1list[1])
  except:
    p1= (kp1list[0])
    p2= (kp1list[1])
      
  for n in range(len(status)):
    x1 , y1 = p1[n][0]-x0, p1[n][1]-y0
    x2 , y2 = p2[n][0]-x0, p2[n][1]-y0

    adx = p2[n][0] - p1[n][0]
    ady = p2[n][1] - p1[n][1]
 
    s3='stroke:rgba(255,0,0,0.2); fill:rgba(255,100,0,0.2);stroke-width:3;'
  
    if status[n] <1:
      nbad+=1
      with open(svgfn, mode='a') as ff:
        # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="'+style[0]+'"  />' )
        ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="2" style="stroke:rgba(255,222,0,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="2" style="stroke:rgba(255,0,222,0.3); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' )
        # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="'+style[5]+'"  />' )
   
      continue
    ngood += 1  
    err= ((x1-x2)**2+(y1-y2)**2)**0.5
    e4=12
    e3 =8
    e2=4
#   # cv2.line(vis, (x1, y1), (x2, y2), green)

    aa=''   
    if err<100:aa+='<circle cx="'+str((x1+x2)/2)+'" cy="'+str((y1+y2)/2)+'" r="'+str(err)+'" style="stroke:rgba(220,220,0,1.0); fill:rgba(222,0,243,1.0);stroke-width:1;"  />'  
    aa+='<line x1="'+str(x1+e2)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1)+'" style="stroke:rgba(255,210,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:2;"  />' 
    # aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+e2)+'" y2="'+str(y1+e4)+'" style="stroke:rgba(255,110,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:1;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2-e4)+'" style="stroke:rgba(20,255,120,0.8);stroke-width:2;"  />' 
    aa+='<line x1="'+str(x2-e2)+'" y1="'+str(y2)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(20,255,120,0.8); stroke-width:2;"  />'  
    # aa+='<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x2)+'" y2="'+str(y2)+'" style="stroke:rgba(222,222,220,0.8); stroke-width:4;"  />'  

    xc = ngood*4+100
    yc =60
    ye=yc-err
 
    if err<100:
      aa+='<circle cx="'+str(xc )+'" cy="'+str(yc )+'" r="'+str(err)+'" style="stroke:rgba(220,222,0,1.0); fill:rgba(220,0,123,0.6);stroke-width:1;"  />'
    # aa+='<circle cx="'+str(xc+xo)+'" cy="'+str(yc+yo)+'" r="3" style="stroke:rgba(220,222,0,1); fill:rgba(220,0,123,0.6);stroke-width:1;"  />'

    with open(svgfn, mode='a') as ff:
        ff.write(aa)
 
      # ff.write('<circle cx="'+str(x2)+'" cy="'+str(y2)+'" r="5" style="'+style[0]+'"  />' )
      # ff.write('<circle cx="'+str(x1)+'" cy="'+str(y1)+'" r="3" style="'+style[2]+'"  />' )
      


      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1)+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="'+str(x1)+'" y1="'+str(y1)+'" x2="'+str(x1+adx )+'" y2="'+str(y1 +ady )+'" style="'+s3+'"  />' )
      # ff.write('<line x1="100" y1="100" x2="'+str(100+p1[n][0]- p2[n][0])+'" y2="'+str(100+p1[n][1]- p2[n][1])+'" style="'+style[2]+'"  />' )



 
 

  footer= '</g>'+'<g transform="translate(1600,0)"><image xlink:href="/F5T/oc/Flight-Imagery/work/processed'+imuri+'/small.jpg" x="0px" y="0px"  height="640px" width="962px"  style=" opacity:1;"/></g>'+'<g transform="translate(0,0)"><text x="10"  y="20" style="font-family: Helvetica; font-size:16;  stroke: rgba(220,110,0,0.0);fill:rgba(0,220,180,0.8);">'+info+'</text>'+'</g></svg>'  
  with open(svgfn, mode='a') as file:
      file.write(footer) 
  print 'oc.plotkpinliers',ngood,nbad, svgfn  ,imuri



def cornqk2abs(corn, qk0):
  px, py, z = oc.qk2pxpy(qk0)
  cout=[]
  for c in range(len(corn)):
    cout.append( [corn[c][0]+px, corn[c][1]+py] )
  return cout  

def cornabs2qk(corn, qk0):
  px, py, z = oc.qk2pxpy(qk0)
  cout=[]
  for c in range(len(corn)):
    cout.append( [corn[c][0]-px, corn[c][1]-py] )
  return cout  



def cornqk2lonlat(corn, qk0):
  px, py, z = oc.qk2pxpy(qk0)
  return cornpxpy2lonlat(cornqk2abs(corn,qk0),z)


def akazematch(img1,img2, AKAZEThreshold=0.0005):
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)
  kp1, desc1 = detector.detectAndCompute(img1, None)
  kp2, desc2 = detector.detectAndCompute(img2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(desc1,desc2)
  p1, p2, kp_pairs = filter_matches2(kp1,kp2, matches)
  H, status = cv2.findHomography(p2, p1, cv2.RANSAC, 8.0)

  print H
  print "inliers", np.sum(status)
  return H


def dcomboc(image, gamma=2.25, ratio=1.0, imname='',a=0.5, b=0.8, c=2.0):
  inv  = 1.0 / gamma
  lutf = np.array([min(1.0, ((  ((i / 255.0) ** gamma) * ratio)** inv ))*255 
    for i in np.arange(0, 256)])

  lut=lutf.astype("uint8")
  p=3
  r=ratio**(a)

  for n in range(100):
    x=1.0*i/255
    oi=r*i
    diff=i-oi
    x2=x**c
    aa=diff*x2*b
    lutf[i]=oi+aa
 

  # .astype("uint8") 
  return cv2.LUT(image, lut)



def dcomboc0(image, gamma=2.25, ratio =1.0, imname='',a=0.5, b=0.8, c=2.0):
  # stops=0
  stops =  round(math.log(ratio**0.5 , 2**0.33333))
  print 'stops', stops
 
  # lutf = np.array([ i   for i in np.arange(0, 256)])
  lutf = np.array([i   for i in np.arange(0, 256)])

  lut=lutf.astype("uint8")
  p=3
  r=ratio**(a)
 
  for i in  range( 256):
    x=1.0*i/255
    oi=r*i
    diff=i-oi
    x2=x**c
    aa=diff*x2*b
    lutf[i]=oi+aa
    # print i, lutf[i], x 

  # lut[255]=255 

  return cv2.LUT(image, lut), lut





def getratio(exif, target=7.1, imuri=''):

  DateTimeOriginal="1900"
  FocalLength=(10,10)
  ExposureTime=(10,10)
  FNumber=(10,10)
  ISOSpeedRatings=0

  for tag, value in exif.items():
    decoded = TAGS.get(tag, tag)
    # if decoded!="MakerNote": print "decoded", decoded, value
    if decoded=="DateTimeOriginal": DateTimeOriginal=value
    if decoded=="FocalLength": FocalLength=value
    if decoded=="ExposureTime": ExposureTime=value
    if decoded=="FNumber": FNumber=value
    if decoded=="ISOSpeedRatings": ISOSpeedRatings=value
    # FocalLength
    # ExposureTime
    # MaxApertureValue
    # ret[decoded] = value
  # print  dirName+"/"+fname, float(FNumber[0])/float(FNumber[1]),(ExposureTime[1])/(ExposureTime[0])

  exdata= imuri+"," +DateTimeOriginal+"," +str(FocalLength[0]/FocalLength[1])+"," +str(float(FNumber[0])/float(FNumber[1])) + ","+str((ExposureTime[1])/(ExposureTime[0]))+","+str(ISOSpeedRatings)
  exdata=   str(FocalLength[0]/FocalLength[1])+"," +str(float(FNumber[0])/float(FNumber[1]))  + ","+str((ExposureTime[1])/(ExposureTime[0]))
  print  "exdata", exdata
  # print imuri, exptime, apert 
  exptime = ExposureTime[1]/ExposureTime[0]
 
  apert =float(FNumber[0])/float(FNumber[1])
  tadj =  exptime/4000.0 
  tadj =  exptime/4000.0 
  if exptime==100:
    tadj=1

  apertadj =  (apert/target)**2
  return {"ratio" : tadj*apertadj, 'aperture':apert ,"shutter" : exptime,"iso" : ISOSpeedRatings}





def histosvg(img, svgfn):
  print 'histosvg', svgfn
  out = ''
  bins = 256
  with open(svgfn, mode='w') as fn:
    fn.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="256" height="256"><g transform="">') 
 
  out += '<line y1="0" x1="0" y2="0" x2="256"  style="stroke:rgba(220,120,0,0.3);stroke-width:4px;"  />' 
  for ax in xrange(257):
    if (ax%32)==0:

      out += '<line y1="'+str(ax)+'" x1="0" y2="'+str(ax)+'" x2="256"  style="stroke:rgba(0,0,0,0.1);stroke-width:1px;"  />' 
    if (ax%64)==0:
      out += '<line y1="'+str(ax)+'" x1="0" y2="'+str(ax)+'" x2="256"  style="stroke:rgba(0,0,0,0.1);stroke-width:2px;"  />' 
    
  histr,binsr = np.histogram(img[:,:,0].flatten(),bins,[0,256])
  histg,binsg = np.histogram(img[:,:,1].flatten(),bins,[0,256])
  histb,binsb = np.histogram(img[:,:,2].flatten(),bins,[0,256])

  ysum = np.sum(histb)
  ysc1 = 10*256.0/ysum
  print 'ysum', ysum, ysc1

  maax = 256.0*256/bins
  h=10
  print 'max', maax
  

  w=768/bins
  w2 = 4
  sb='stroke:rgba(0,0,0,0.6); fill:rgba(255,100,0,0.2);stroke-width:20;'

  s0='stroke:rgba(0,0,255,0.8); fill:rgba(255,100,0,0.2);stroke-width:'+str(w2)+';'
  s1='stroke:rgba(0,200,80,0.8); fill:rgba(255,100,0,0.2);stroke-width:'+str(w2)+';'
  s2='stroke:rgba(255,0,0,0.8); fill:rgba(255,100,0,0.2);stroke-width:'+str(w2)+';'

  pb=''
  pg=''
  pr=''

  for b in range(bins):
    # print b, hist0[b], hist1[b], hist2[b]
    x1=256-b
    x2=x1
    y10=0
 
    y20=(histr[b]*ysc1)
    y21=(histg[b]*ysc1) 
    y22=(histb[b]*ysc1) 
 
    # out += '<line x1="'+str(x1+sb)+'" y1="'+str(y10)+'" x2="'+str(x2+sb)+'" y2="'+str(y20)+'" style="'+s0+'"  />' 
    # out += '<line x1="'+str(x1+sg)+'" y1="'+str(y11)+'" x2="'+str(x2+sg)+'" y2="'+str(y21)+'" style="'+s1+'"  />' 
    # out += '<line x1="'+str(x1+sr)+'" y1="'+str(y12)+'" x2="'+str(x2+sr)+'" y2="'+str(y22)+'" style="'+s2+'"  />'
    pb+=' '+str(y20)+','+str(x2) 
    pg+=' '+str(y21)+','+str(x2) 
    pr+=' '+str(y22)+','+str(x2) 
    # out += '<line x1="'+str(x1)+'" y1="'+str(y10)+'" x2="'+str(x2)+'" y2="'+str(y10+1)+'" style="'+sb+'"  />' 
    # out += '<line x1="'+str(x1)+'" y1="'+str(y11)+'" x2="'+str(x2)+'" y2="'+str(y11+1)+'" style="'+sb+'"  />' 
    # out += '<line x1="'+str(x1)+'" y1="'+str(y12)+'" x2="'+str(x2)+'" y2="'+str(y12+1)+'" style="'+sb+'"  />' 
  pb+=' '+str(x2+0.02)+','+str(y10)  + ' 0,256 ' 
  out += '<polygon points="'+pb+'"  style="fill:rgba(0,0,255,0.2); stroke:rgba(0,5,250,0.8);stroke-width:1px;"  />' 
  pg+=' '+str(x2+0.02)+','+str(y10)   + ' 0,256 '   
  out += '<polygon points="'+pg+'"  style="fill:rgba(0,255,0,0.2); stroke:rgba(0,250,0,0.8);stroke-width:1px;"  />' 
  pr+=' '+str(x2+0.02)+','+str(y10)    + ' 0,256 '  
  out += '<polygon points="'+pr+'"  style="fill:rgba(255,0,0,0.2); stroke:rgba(250,0,0,0.8);stroke-width:1px;"  />' 

  with open(svgfn, mode='a') as fn:
    fn.write(out) 
    fn.write('</g></svg>') 







def plotlut(lut, svgfn='lut.svg', tex="lut"):
    print 'plotlut', svgfn
    bins = 256
    out=''
    with open(svgfn, mode='w') as fn:
      fn.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="276" height="276"><g transform="scale(1.0)"><g transform="translate(10,10)">') 
      fn.write('<polygon points="0,0 0,256 256,256 256,0 "  style="fill:rgba(122,122,122,0.1); stroke:rgba(40,45,44,0.4);stroke-width:0.2px;"  />' )
   

    sb='stroke:rgba(0,0,0,0.6); fill:rgba(255,100,0,0.2);stroke-width:20;'

    x0=0
    y0=256

    pb=' '+str(x0)+','+str(y0) 
    for b in range(bins):
      
      out += '<circle cx="'+str(b+x0)+'"  cy="'+str(y0-lut[b])+'" r="2"  style="fill:rgba(0,0,255,0.0); stroke:rgba(220,5,250,0.8);stroke-width:0.5px;"  />\n'
      pb+=' '+str(b+x0)+','+str(y0-lut[b])  
      # x=b*1.0/256
      # tval = tfunc(x) *256 
      # out += '<circle cx="'+str(b+x0)+'"  cy="'+str(y0-tval)+'" r="2"  style="fill:rgba(0,0,255,0.0); stroke:rgba(220,5,0,0.8);stroke-width:0.5px;"  />\n'
      # print b, lut[b],x, tval
    pb+=' '+str(255+x0)+','+str(y0)   
    # out += '<polygon points="'+pb+'"  style="fill:rgba(0,0,255,0.2); stroke:rgba(0,5,250,0.8);stroke-width:2px;"  />' 
    out+='<text x="50" y="150" font-family="Verdana" font-size="20">'+tex+'</text>'

    with open(svgfn, mode='a') as fn:
      fn.write(out) 
      fn.write('</g></g></svg>') 




def lonlat2pxpy(lng,lat,  zoom):
    """Given two floats and an int, return a 2-tuple of ints.
    Note that the pixel coordinates are tied to the entire map, not to the map
    section currently in view.
    """
    # print 'lonlat2pxpy',lng,lat,  zoom
    # assert isinstance(lat, (float, int, long)), \
    #     ValueError("lat must be a float")
    # lat = float(lat)
    # assert isinstance(lng, (float, int, long)), \
    #     ValueError("lng must be a float")
    # lng = float(lng)
    # assert isinstance(zoom, int), TypeError("zoom must be an int from 0 to 30")
    # assert 0 <= zoom <= 30, ValueError("zoom must be an int from 0 to 30")
    # print lng +0.000000001
    # if not isinstance(lng, float):
    #   print 'lonlat2pxpy problem', lng, lat
 
       
    cbk = CBK[zoom]

    x = ( (cbk + (lng * CEK[zoom])))

    foo = math.sin(lat * math.pi / 180)
    if foo < -0.9999:
        foo = -0.9999
    elif foo > 0.9999:
        foo = 0.9999

    y = ( (cbk + (0.5 * math.log((1+foo)/(1-foo)) * (-CFK[zoom]))))

    return x, y





def lonlat2tilexy(lng,lat,  zoom):
    """Given two floats and an int, return a 2-tuple of ints.
    Note that the pixel coordinates are tied to the entire map, not to the map
    section currently in view.
    """
    assert isinstance(lat, (float, int, long)), \
        ValueError("lat must be a float")
    lat = float(lat)
    assert isinstance(lng, (float, int, long)), \
        ValueError("lng must be a float")
    lng = float(lng)
    assert isinstance(zoom, int), TypeError("zoom must be an int from 0 to 30")
    assert 0 <= zoom <= 30, ValueError("zoom must be an int from 0 to 30")

    cbk = CBK[zoom]

    x = ( (cbk + (lng * CEK[zoom])))

    foo = math.sin(lat * math.pi / 180)
    if foo < -0.9999:
        foo = -0.9999
    elif foo > 0.9999:
        foo = 0.9999

    y = ( (cbk + (0.5 * math.log((1+foo)/(1-foo)) * (-CFK[zoom]))))

    return x/256, y/256





def pxpy2lonlat(x, y, zoom):
    """Given three ints, return a 2-tuple of floats.
    Note that the pixel coordinates are tied to the entire map, not to the map
    section currently in view.
    """
    # assert isinstance(x, (int, long)), \
    #     ValueError("px must be a 2-tuple of ints")
    # assert isinstance(y, (int, long)), \
    #     ValueError("px must be a 2-tuple of ints")
    # assert isinstance(zoom, int), TypeError("zoom must be an int from 0 to 30")
    # assert 0 <= zoom <= 30, ValueError("zoom must be an int from 0 to 30")

    foo = CBK[zoom]
    lng = (x - foo) / CEK[zoom]
    bar = (y - foo) / -CFK[zoom]
    blam = 2 * math.atan(math.exp(bar)) - math.pi / 2
    lat = blam / (math.pi / 180)

    return  lng, lat  





def callgcutimuri(imuri, zout, mixdir = "/home/oc/F5T/oc/Flight-Imagery/work/tiles/20151121mixmf/", basedir = "/home/oc/F5T/oc/Flight-Imagery/work/tiles/20151121mixmf/",vedir = '/home/oc/F5T/oc/Flight-Imagery/work/ve/', noseam=True, limits = [2000,2000,8500,8500], ssdo=1):

  nos=""
  if noseam: nos="noseam"
  tb=oc.tnow()
  jr = 'http://25.59.64.230/repo/getstatus.php?z='+str(zout)+'&stage=georefdone&imuri='+imuri

  r = requests.get(jr)
  print 'callgcutimuri',r.url
  print r.text




  b = json.loads(r.text)
  try:
     ddd =b['imuri']
  except:
    print 'callgcutjs json no good', imuri
    return   

  # quit()
  # b = {"imurib": "", "run": "77/", "imuri": "20151019/wf00/100D3200/DSC_0060.JPG", "corrected": [[-122.23361313343048, 38.2141341797837], [-122.23070025444031, 38.229584166202926], [-122.21797585487366, 38.22751930070061], [-122.22091019153595, 38.2129961623591]], "coords": [[-50.85024642944336, -271.1988525390625], [905.6668090820312, -246.8851776123047], [892.3177490234375, 361.6447448730469], [-56.98851776123047, 345.4422302246094]], "stage": "relabdone", "orig": [[0, 0], [963, 0], [963, 640], [0, 640]]
  a={}  
  
  a["imuri"] = b["imuri"]
  a["coords"] = b["coords"]
  if  b["coords"][0][0]>-120 or  b["coords"][0][0]<-123 or  b["coords"][1][0]>-120 or  b["coords"][1][0]<-123 :
    a["coords"] = b["corrected"]
  a['a']=0.5
  a['b']=0.8
  a['target']=7.6
  a['zwork']=zout
  a['mixdir']=mixdir
  a['basedir']=basedir
  a['vedir']=vedir
  a['minwidth']=limits[0]
  a['minheight']=limits[1]  
  a['maxwidth']=limits[2]
  a['maxheight']=limits[3]
  a['noseam']=nos # seam or not
  a['ssdo']=ssdo # save or not

  
  encoded = oc.b64encode(oc.jdumps(a))
  cmd = "./gcutjs "+encoded

  th=oc.thuman()
  with open('callgcutjsblend.log.txt', mode='a') as fn:
    fn.write( th + ' ' + cmd + '\n') 
  oc.system(cmd)
  te=oc.tnow()
  oc.coutb( 'gcutjs '+ "%0.3f"%(te-tb) )




def testcallgcutjsimuri():
  jsonin='{"imuri":"/20151121/wf00/100D3200/DSC_0100.JPG","stage":"georefdone","substage":"gdir","status":"test0201wf","coords":[[-122.381859093,37.736910135],[-122.40041209,37.7375502874],[-122.399523782,37.7474387635],[-122.381264523,37.7464597942]],"clon":-122.390764872,"clat":37.742089745,"time":1459141221.52,"timetaken":9.66013383865,"inliers":1826,"area":0.00017949064340428,"matches":2947,"th":"2016-03-27 22:00:21 -07:00"}'
  imuri = "/20151121/wf00/100D3200/DSC_0100.JPG"
  zout=18
  mixdir = 'testmix'
  urlprefix = "http://10.0.0.4/F08/FI/"
  callgcutjsimuri(imuri, zout, mixdir = mixdir, basedir = mixdir,vedir = mixdir, noseam=True, limits = [2000,2000,8500,8500], ssdo=1, jsonin=jsonin,  urlprefix = urlprefix)




def callgcutjsimuri(imuri, zout, mixdir = "/home/oc/F5T/oc/Flight-Imagery/work/tiles/20151121mixmf/", basedir = "/home/oc/F5T/oc/Flight-Imagery/work/tiles/20151121mixmf/",vedir = '/home/oc/F5T/oc/Flight-Imagery/work/ve/', noseam=True, limits = [2000,2000,8500,8500], ssdo=1, jsonin="", urlprefix = "http://10.0.0.4/F08/FI/"):

  nos=""
  if noseam: nos="noseam"
  tb=oc.tnow()
  # jr = 'http://25.59.64.230/repo/getstatus.php?z='+str(zout)+'&stage=georefdone&imuri='+imuri

  # r = requests.get(jr)
  # print 'callgcutimuri',r.url
  # print r.text




  b = json.loads(jsonin)
  try:
     ddd =b['imuri']
  except:
    print 'callgcutjs json no good', imuri
    return   

  # quit()
  # b = {"imurib": "", "run": "77/", "imuri": "20151019/wf00/100D3200/DSC_0060.JPG", "corrected": [[-122.23361313343048, 38.2141341797837], [-122.23070025444031, 38.229584166202926], [-122.21797585487366, 38.22751930070061], [-122.22091019153595, 38.2129961623591]], "coords": [[-50.85024642944336, -271.1988525390625], [905.6668090820312, -246.8851776123047], [892.3177490234375, 361.6447448730469], [-56.98851776123047, 345.4422302246094]], "stage": "relabdone", "orig": [[0, 0], [963, 0], [963, 640], [0, 640]]
  a={}  
  
  a["imuri"] = b["imuri"]
  a["coords"] = b["coords"]
  if  b["coords"][0][0]>-120 or  b["coords"][0][0]<-123 or  b["coords"][1][0]>-120 or  b["coords"][1][0]<-123 :
    a["coords"] = b["corrected"]
  a['a']=0.5
  a['b']=0.8
  a['target']=7.6
  a['zwork']=zout
  a['mixdir']=mixdir
  a['basedir']=basedir
  a['vedir']=vedir
  a['minwidth']=limits[0]
  a['minheight']=limits[1]  
  a['maxwidth']=limits[2]
  a['maxheight']=limits[3]
  a['noseam']=nos # seam or not
  a['ssdo']=ssdo # save or not
  a['ssdo']=ssdo # save or not
  a["urlprefix"] = urlprefix

  
  encoded = oc.b64encode(oc.jdumps(a))
  cmd = "./gcutjs "+encoded

  th=oc.thuman()
  with open('callgcutjsblend.log.txt', mode='a') as fn:
    fn.write( th + ' ' + cmd + '\n') 
  oc.system(cmd)
  te=oc.tnow()
  oc.coutb( 'gcutjs '+ "%0.3f"%(te-tb) )




def callgcutjs(js , zout, mixdir, noseam=True, limits = [2000,2000,8500,8500]):
  tb=oc.tnow()
  b = js 
  try:
     ddd =b['imuri']
  except:
    print 'callgcutjs json no good' 
    return   


  a={}  
  
  a["imuri"] = b["imuri"]
  a["coords"] = b["coords"]
  if  b["coords"][0][0]>-120 or  b["coords"][0][0]<-123 or  b["coords"][1][0]>-120 or  b["coords"][1][0]<-123 :
    a["coords"] = b["corrected"]
  a['a']=0.5
  a['b']=0.8
  a['target']=7.6
  a['zwork']=zout
  a['mixdir']=mixdir
  a['minwidth']=limits[0]
  a['minheight']=limits[1]  
  a['maxwidth']=limits[2]
  a['maxheight']=limits[3]
  encoded = oc.b64encode(oc.jdumps(a))
  cmd = "./gcutjs "+encoded

  th=oc.thuman()
  with open('callgcutjsblend.log.txt', mode='a') as fn:
    fn.write( th + ' ' + cmd + '\n') 
  # oc.system(cmd)
  te=oc.tnow()
  oc.coutb( 'gcutjs '+ "%0.3f"%(te-tb) )


def shrink(mixdir):
  basein=mixdir
  baseout=mixdir

  oc.smartshrink(basein,baseout, zoomin=zout)
def tshrink(mixdir, rawfn ="qklistraw.txt", minzoom = 10,zoomin=18):
# def tshrink(mixdir, rawfn ="qklistraw.txt", minzoom = 10,zoomin=18):
  if not os.path.isfile(rawfn ):
    print 'tshrink '+rawfn+' not found'
    return
  oc.qklistshrink(rawfn, mixdir,mixdir, minzoom = minzoom,zoomin=24)
  tst=oc.tstr()
  shutil.move(rawfn,'qklist'+tst+'.txt')
  


 







def corn2svg(corners,names=[],  zoom=14, svgfn='corn2svg.svg' ):
  qklist = corna2qklist(corners, zoom)
  svgout=''

  tx0, ty0, z=qk2xy(qklist[0] )  
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx
      dya=ty
      # print qk
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str( (dxa-tx0)*256)+'px" y="'+str((dya-ty0)*256)+'px" height="256px" width="256px" style=" opacity:1;"   dxa="'+str(dxa)+'" />';
 
  x0,y0 =  tx0*256, ty0*256    
  x1,y1=0,0
  npp=0
  for ccc in range(1):
    p=''
    # print pol1
    try:
      nt =  names[npp]
    except:
      nt=str(npp)
      
    npp+=1 
    xavg=0
    yavg=0
    for c in corners:
      px, py = lonlat2pxpy(c[0],c[1],zoom) 
      x1, y1 = px-x0, py-y0

      xavg=xavg+x1/4
      yavg=yavg+y1/4
      p += " "+str(x1)+","+str(y1)+""
      tt =  "%0.4f"%(c[0])+','+"%0.4f"%(c[1])
      svgout+='<text  text-anchor="middle"  x="'+"%0.4f"%(x1)+'" y="'+"%0.4f"%(y1)+'" style="font-family: helvetica, sans-serif; font-weight: normal; font-style: normal" font-size="10px" fill="#0ff">'+tt+'</text>' 
    svgout+= '<polygon points="'+p+'"  style="fill:rgba(255,0,0,0.1); stroke:rgba(225,250,30,0.8);stroke-width:1px;"  />'
    svgout += '<text  text-anchor="middle" x="'+"%0.4f"%(xavg)+'" y="'+"%0.4f"%(yavg-20)+'" style="font-family: helvetica, sans-serif; font-weight: normal; font-style: normal" font-size="12px" fill="#ff0">'+nt+'</text>' 
    svgout += '<circle cx="'+str(xavg )+'" cy="'+str(yavg)+'" r="12" style="stroke:rgba(0,110,220,0.9); fill:rgba(0,255,0,0.9);stroke-width:2;"  />'
 
 


  svgo='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform=" translate(0,0)">' + svgout + '</g></svg>'
  with open(svgfn, mode='w') as ff:
        ff.write(svgo ) 
  coutb('corna2svg wrote '+ svgfn)         








def corna2svg(corners,names=[],  zoom='auto', svgfn='corna2svg.svg' ,scale=1):
  qklist14 = corna2qklist(corners, 14)
  qklist16 = corna2qklist(corners, 16)
  qklist18 = corna2qklist(corners, 18)
  qklist =qklist18
  z = 18
  if len(qklist) > 120:
    qklist =qklist16
    z = 16
  if len(qklist) > 120:
    qklist =qklist14
    z = 14

  if zoom !='auto':
    qklist  = corna2qklist(corners, zoom)
    z = zoom
      


  svgout=''

  tx0, ty0, z=qk2xy(qklist[0] )  
  for qk in qklist:

      tx, ty, z=qk2xy(qk )
      dxa=tx
      dya=ty
      # print qk
      svgout+='<image xlink:href="/F5T/oc/Flight-Imagery/work/ve/a'+qk+'.jpg" x="'+str( (dxa-tx0)*256)+'px" y="'+str((dya-ty0)*256)+'px" height="256px" width="256px" style=" opacity:1;"   dxa="'+str(dxa)+'" />';
 
  x0,y0 =  tx0*256, ty0*256    
  x1,y1=0,0
  npp=0
  for pol1 in corners:
    p=''
    # print pol1
    try:
      nt =  names[npp]
    except:
      nt=str(npp)
      
    npp+=1 
    xavg=0
    yavg=0
    for c in pol1:
      px, py = lonlat2pxpy(c[0],c[1],z) 
      x1, y1 = px-x0, py-y0

      xavg=xavg+x1/4
      yavg=yavg+y1/4
      p += " "+str(x1)+","+str(y1)+""
      tt =  "%0.4f"%(c[0])+','+"%0.4f"%(c[1])
      svgout+='<text  text-anchor="middle"  x="'+"%0.4f"%(x1)+'" y="'+"%0.4f"%(y1)+'" style="font-family: helvetica, sans-serif; font-weight: normal; font-style: normal" font-size="10px" fill="#0ff">'+tt+'</text>' 
    svgout+= '<polygon points="'+p+'"  style="fill:rgba(255,0,0,0.1); stroke:rgba(225,250,30,0.8);stroke-width:1px;"  />'
    svgout += '<text  text-anchor="middle" x="'+"%0.4f"%(xavg)+'" y="'+"%0.4f"%(yavg-20)+'" style="font-family: helvetica, sans-serif; font-weight: normal; font-style: normal" font-size="12px" fill="#ff0">'+nt+'</text>' 
    svgout += '<circle cx="'+str(xavg )+'" cy="'+str(yavg)+'" r="12" style="stroke:rgba(0,110,220,0.9); fill:rgba(0,255,0,0.9);stroke-width:2;"  />'
 
 


  svgo='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4000" height="4000"><g transform=" scale('+str(scale)+')">' + svgout + '</g></svg>'
  with open(svgfn, mode='w') as ff:
        ff.write(svgo ) 
  coutb('corna2svg wrote '+ svgfn)         

def getallmf(imuri):

  rr =  imuri.split('/')

  rw= '/'+rr[1]+'/wf00/'+rr[3]+'/'+rr[4]
  r0= '/'+rr[1]+'/mf00/'+rr[3]+'/'+rr[4]
  r1= '/'+rr[1]+'/mf01/'+rr[3]+'/'+rr[4]
  r2= '/'+rr[1]+'/mf02/'+rr[3]+'/'+rr[4]
  r3= '/'+rr[1]+'/mf03/'+rr[3]+'/'+rr[4]

  return [rw, r0, r1, r2,r3]
 


def pil2cv2(im):

  open_cv_image = np.array(im) 
  open_cv_image = open_cv_image[:, :, ::-1].copy() 

  return open_cv_image


def imuri2imnum(imuri):
  # /20150815/wf00/100D3200/DSC_0053.JPG
  ix,iy = imuri2ixiy(imuri)
  imnum = ix+iy*1000
  # print 'imuri2imnum',imnum
  return imnum


def imuri2ixiy(imuri):
  # /20150815/wf00/100D3200/DSC_0053.JPG
  ix = int(imuri[-8:-4])
  iy = int(imuri[-20:-18])

  return ix,iy
 
def getnext(imuri, step = 1):

  imnum = imuri2imnum(imuri) + step
  if imnum < 1: imnum = 1
  ix,iy = oc.imnum(imnum)

  out = imuri[:-20]+"%02d"%iy+imuri[-18:-8]+"%04d"%ix+imuri[-4:]

  return out
 
def getimuriloc(imuri):
  
  loc = 'http://25.113.154.8/F7T/Flight-Imagery'+imuri
  if imuri[1:9] == "20140513":
    loc = 'http://25.59.64.230/F5T/oc/Flight-Imagery'+imuri

  
  return loc



def getdateloc(date):
  
  loc = 'http://25.113.154.8/F7T/Flight-Imagery/'+date+"/"
  
  return loc


# def getimuri(imuri):
#   print 'getimuri',imuri

  
#   except:
#     coutr("getimuri bad image")
#     return -1
#   return ima  

def getimg(imurl):
  # print 'getimg',imurl
   
  r = requests.get( imurl, stream=True)
  if r.status_code == 200:
        # r.raw.decode_content = True
        # featdeca = r.content
        a=0
  # ima = Image.open('smalltmp.jpg')
  else:
    print "not found"
    coutr("getimg not found " + imurl)
    return -1

 
  try:
    ima = Image.open(StringIO(r.content) )
  except:
    coutr("getimuri bad image")
    return -1
  return ima  



def makeimuri(date,mf,num):
  ix,iy = oc.imnum(num)
  return   '/'+date+'/'+mf+'/1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG'
   
def getimuri(date,mf,num):
  ix,iy = oc.imnum(num)
  return   '/'+date+'/'+mf+'/1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG'
   
   


def georefsingle(imuri, zoom=18, update = True, veroot='/home/oc/F5T/oc/Flight-Imagery/work/ve/', tilesource='ve' ):
 
  rs = requests.get('http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/getcenter.php?imuri='+imuri)
  print 'http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/getcenter.php?imuri='+imuri
 
  jjs = json.loads(rs.text)
  # print "georefsingle corns"
  # quit()
 
  try:
    qklist = oc.cornqklist(jjs[imuri], zoom)
  except:
    print 'georefsingle center not found', rs.url
    return 'pcenter'  

  if len(qklist)<16:
    oc.couty("georefsingle pqklist  < 16 tiles, skipping")
    return 
  if len(qklist)>400:
    oc.couty("georefsingle pqklist  > 400 tiles, skipping")
    return 
  # qkmont = oc.qkmontage(qklist,ve=veroot )

  # qkmont.save('qkmonts.jpg','JPEG')

  # print 'georefsingle qkmonts.jpg' 
 
  imc = oc.getimgsize(imuri)
  # print  "georefsingle imc", imc
  k2a, ddesca = oc.getfeatimuri(imuri)
  if len(k2a)<100:
    a = oc.dncimuri(imuri)
    k2a, ddesca = a['kp'], a['desc']
  if len(k2a)<100:
    print "pkp"
 
  # k2b, ddescb = oc.mosaicfeaturesabs(qklist)
  # k2b, ddescb = oc.mosaicfeatures(qklist)

  print 'georefsingle montagefeatures start'

  k2b, ddescb = oc.montagefeatures(qklist,tilesource = tilesource, ext='.jpg', montagesave="georefsinglemontage.jpg")

  print 'georefsingle len',len(k2b)
  if len(k2b) < 500:
    return "pkp"

  # print 'pt',k2b[0].pt
  # oc.plotkps(k2b,   svgfn='kpslonlat.svg', qklist=qklist   )
 

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(ddesca,ddescb)
  p1, p2, kp_pairs = oc.filter_matches2(k2a, k2b, matches)
  Hpx2pxqk, status = cv2.findHomography(p1, p2, cv2.RANSAC, 8.0)
  # print Hpx2pxqk
  kpabs, limits =  oc.xformkp(k2a, Hpx2pxqk)
  p1qk  =  oc.xformp1(p1, Hpx2pxqk)
  cornpxqk = np.float32(cv2.perspectiveTransform(imc.reshape(1, -1, 2), Hpx2pxqk ).reshape(-1, 2) )
 
  cornlonlat = oc.cornqk2lonlat(cornpxqk, qklist[0])
  print 'cornlonlat',cornlonlat
  print 'georefsingle',len(k2a),'to',len(k2b)
  inliers = int( np.sum(status))
  print(Fore.WHITE  + Back.BLUE +' inliers single: ' + str(inliers) +' ' + Style.RESET_ALL)
  
  print 'cornlonlat', cornlonlat

  # oc.plotkpinliersv1(kp1list=[p1qk,p2],status=status, x0=0, y0=0,svgfn='kplonlat.svg', qklist=qklist, corners= [cornlonlat, cornpxqk], imuri=imuri, scale=0.5)  
  te=oc.tnow()
  th =oc.thuman()
  payloado =  {"imuri": imuri , "stage": "georefdone",   "coords": cornlonlat  ,"time": te,"th":th ,"inliers": inliers, "status" :'georefsingle'}
  print payloado 
  payload= json.dumps(payloado) 
  if update:
    r = requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/insert.php?js="+payload)
    print r.url


  return 'ok'


def qklistbounds(qklist):

  minlon, minlat = 180,90
  maxlon, maxlat = -180,-90

  for qk in qklist:
    lon,lat= oc.qk2lonlat(qk)
    lonbr, latbr=  oc.qk2lonlat(oc.qk2qk(qk,1,1))
    minlon = min(minlon,lon)
    minlat = min(minlat,latbr)
    maxlon = max(maxlon,lonbr)
    maxlat = max(maxlat,lat)

  return [[minlon,minlat], [maxlon, minlat],  [maxlon, maxlat], [minlon, maxlat]]  


  # if True:
  #   kpabs, limits =  xformkp(k2a, Hpx2px18)
  #   p1abs, limits = xformp1(p1, Hpx2px18)
  #   p1d, p2d, kp_pairs = oc.dist_filter_matches(kpabs, k2b, matches, dist=200)
  #   px0, py0, z = oc.qk2pxpy(qklist[0])
   
  #   p1a = oc.translatep1(p1d, -px0, -py0)
  #   p2a = oc.translatep1(p2d, -px0, -py0)
   
  #   Hrefinepx182px18, status2 = cv2.findHomography(p1a, p2a, cv2.RANSAC, 8.0)
  #   print 'inliers2', np.sum(status2)

  # cpx18 = np.float32(cv2.perspectiveTransform(imc.reshape(1, -1, 2), Hpx2px18 ).reshape(-1, 2) )
  # Hpx2lonlat =   cv2.getPerspectiveTransform( np.float32(imc), np.float32(oc.coordspx2lonlat(c2)))
 

  # c2refinepx18 = np.float32(cv2.perspectiveTransform(cpx18, Hrefinepx182px18 ).reshape(-1, 2) )
  # H2px2lonlat =   cv2.getPerspectiveTransform( np.float32(imc), np.float32(oc.coordspx2lonlat( c2refinepx18)))
  # c2geo = cv2.perspectiveTransform(imc.reshape(1, -1, 2), Hpx2geo ).reshape(-1, 2)
  # c2geo2 = cv2.perspectiveTransform(imc.reshape(1, -1, 2), H2px2geo ).reshape(-1, 2)
  # # oc.plotkps(kpgeo, kp2=k2b, status=status,  x0=0, y0=0,svgfn='kpsingle.svg', qklist=qklist, corners= [oc.coordspx2lonlat(c2)], imuri='',scale=1.0)
  # oc.plotkpinliersv1(kp1list=[p2,p1abs],status=status, x0=0, y0=0,svgfn='kpsingle.svg', qklist=qklist, corners= [c2geo], imuri='')  
  # oc.plotkpinliersv1(kp1list=[p1a,p2a],status=status2, x0=0, y0=0,svgfn='kpsingle2.svg', qklist=qklist, corners= [c2geo2], imuri='')  
  # return {'imuri':imuri, 'kp':k2a, 'ddesc':ddesca, 'imc': imc, 'pxcorn': c2, 'coords': c2geo, 'inliers':np.sum(status) } 
 

def getchildtiles(qk,z):
  qkz=len(qk)
  tx0,ty0,z0=oc.qk2xy(qk)
  if z<=z0:
    return [qk[:z]]
  # print tx0,ty0,z0
  mult = 2**(z-z0)

  # print 'getchildtiles z,z0',z, z0
  # print 'getchildtiles mult',mult
  txrange = xrange(tx0*mult,(tx0+1)*mult )
  tyrange = xrange(ty0*mult,(ty0+1)*mult )
  # print txrange
  qklist=[]
  for y in tyrange:
    for x in txrange:
      qklist.append(oc.xy2qk(x,y, z))

  return qklist


 

def getsize(filename):
    """Return the size of a file, reported by os.stat()."""
    if not os.path.isfile(filename):
      return -1
    return os.stat(filename).st_size





def rmtiles(qk):
  
  #  remove delete move plain z files
  
  for z in range(12,22):
    qklist = oc.getchildtiles(a0, z)
    print z,'qklist',len(qklist)

    dellist = {}
    for qk in qklist:
      fn = '/home/oc/F5T/oc/Flight-Imagery/work/tiles/20150124mixmf/z'+qk+'.jpg'
      size = getsize(fn)
      if size==2187:
        dellist[fn] = size
        dest = '/home/oc/F5T/oc/Flight-Imagery/work/tiles/20150124mixmf/junk/z'+qk+'.jpg'
        shutil.move(fn,dest) 
        print 'moved', qk



def counttiles(poly,zoom):
  clim = oc.cornabox([poly])

  tx0, ty0= lonlat2xy(clim[0][0],clim[0][1]  , zoom)  
  tx1, ty1= lonlat2xy(clim[2][0],clim[2][1]  , zoom)
  print clim  
 
 
  height=ty1-ty0+1
  width=tx1-tx0+1

  return height*width




def getimuritiles(imuri,zout, svgout='getimuritiles.svg'):
  """
  find tiles inside imuri poly
  """
  jr = 'http://25.59.64.230/repo/getstatus.php?z='+str(zout)+'&stage=georefdone&imuri='+imuri

  r = requests.get(jr)
  print 'getimuritiles',r.url
 
  b = json.loads(r.text)
  try:
     ddd =b['imuri']
  except:
    print 'callgcutjs json no good', imuri
    return  

  print b['coords']
  qklist = oc.corna2qklist([b['coords']], zout)   
  # print qklist

  cc = [ oc.cornabox(cornersa=[b['coords']]) ]
  
  print 'getimuritiles len cc',len(cc)
  cornstyle = {}
  cornstyle['limits']= {'cc':oc.cornabox(cornersa=[b['coords']]), 'style': 'stroke:rgba(255,0,0,0.7);fill:rgba(255,255,0,0.0);  stroke-width:10;'}
  cornstyle['a'] =  {'cc':b['coords'], 'style': 'stroke:rgba(255,0,0,0.7);;fill:rgba(0,0,250,0.2);  stroke-width:2;'}
  # cornstyle['b']={'cc':cc, 'style': 'stroke:rgba(255,0,0,0.7); stroke-width:1;'}
  out={}
  for qk in qklist:

    incount = len(oc.corninpoly(b['coords'],oc.qklistbounds([qk])))
    out[qk] = incount
    # print "incount",qk, incount
    ss = 'stroke:rgba(25,25,25,0.3);fill:rgba(0,0,0,0.1); stroke-width:1;'
    if incount==1:ss = 'stroke:rgba(255,0,0,0.3);fill:rgba(155,0,0,0.1); stroke-width:1;'
    if incount==2:ss = 'stroke:rgba(255,255,0,0.3);fill:rgba(255,0,0,0.1); stroke-width:1;'
    if incount==3:ss = 'stroke:rgba(0,255,0,0.3);fill:rgba(255,255,0,0.1); stroke-width:1;'
    if incount==4:ss = 'stroke:rgba(0,255,250,0.8);fill:rgba(255,0,255,0.1); stroke-width:1;'


    cornstyle[qk]={'cc':oc.qklistbounds([qk]), 'style':ss ,"cornersin":incount}

  if len(svgout)>3:
    oc.plotquads( cornstyle=cornstyle, svgfn=svgout , tilesource= '/F5T/oc/Flight-Imagery/work/ve/',zoom=16  ,scale=1.0 )

  return out        




jjcounties=[{"type":"Feature","bbox":[-122.3296,37.4548,-121.4697,37.9040],"properties":{"kind":"county","name":"Alameda","state":"ca","center":[-121.8996,37.6797],"centroid":[-121.8902,37.6466]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-122.2693,37.9040],[-122.1872,37.8218],[-122.0448,37.7999],[-121.9626,37.7177],[-121.5573,37.8218],[-121.5573,37.5425],[-121.4697,37.4822],[-121.4752,37.4822],[-121.8531,37.4822],[-121.9243,37.4548],[-122.0831,37.4768],[-122.1160,37.5041],[-122.1653,37.6684],[-122.2474,37.7232],[-122.2419,37.7561],[-122.3296,37.7835],[-122.3131,37.8985]]]]}},
{"type":"Feature","bbox":[-122.4281,37.7177,-121.5354,38.1011],"properties":{"kind":"county","name":"Contra Costa","state":"ca","center":[-121.9818,37.9097],"centroid":[-121.9266,37.9207]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-121.6285,38.1011],[-121.5792,38.0957],[-121.5573,37.9259],[-121.5792,37.8601],[-121.5354,37.8492],[-121.5573,37.8218],[-121.9626,37.7177],[-122.0448,37.7999],[-122.1872,37.8218],[-122.2693,37.9040],[-122.3131,37.8985],[-122.3843,37.9094],[-122.4281,37.9642],[-122.3679,37.9806],[-122.3624,38.0135],[-122.3022,38.0135],[-122.2693,38.0628],[-122.1488,38.0299],[-121.8640,38.0683],[-121.7764,38.0190]]]]}},
{"type":"Feature","bbox":[-122.6308,38.1559,-122.0612,38.8624],"properties":{"kind":"county","name":"Napa","state":"ca","center":[-122.3460,38.5100],"centroid":[-122.3327,38.5068]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-122.3953,38.8624],[-122.2857,38.8405],[-122.1050,38.5119],[-122.1269,38.4243],[-122.0612,38.3257],[-122.2036,38.3147],[-122.1981,38.1559],[-122.4062,38.1559],[-122.3515,38.1942],[-122.3679,38.2490],[-122.4993,38.4243],[-122.4829,38.4517],[-122.6308,38.5721],[-122.6253,38.6653],[-122.4665,38.7036],[-122.3789,38.8022]]]]}},
{"type":"Feature","bbox":[-122.5158,37.7068,-122.3569,37.8108],"properties":{"kind":"county","name":"San Francisco","state":"ca","center":[-122.4364,37.7588],"centroid":[-122.4409,37.7517]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-122.4281,37.7068],[-122.5048,37.7068],[-122.5148,37.7168],[-122.52,37.7835],[-122.4862,37.8108],[-122.3862,37.8208],[-122.3469,37.7087],[-122.3898,37.7068]]]]}},
{"type":"Feature","bbox":[-122.5158,37.1098,-122.0831,37.7068],"properties":{"kind":"county","name":"San Mateo","state":"ca","center":[-122.2994,37.4089],"centroid":[-122.3253,37.4272]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-122.4281,37.7068],[-122.3898,37.7068],[-122.37,37.695],[-122.3643,37.6356],[-122.3269,37.6137],[-122.1160,37.5041],[-122.0831,37.4768],[-122.1926,37.4329],[-122.1926,37.3179],[-122.1543,37.2851],[-122.1543,37.2139],[-122.3186,37.1865],[-122.2912,37.1098],   [-122.3412,37.1098],[-122.4162,37.1974],[-122.4262,37.2274],[-122.4208,37.3617],[-122.4601,37.4822],[-122.5358,37.4906],[-122.52,37.6868],[-122.5048,37.7068]]]]}}, 
{"type":"Feature","bbox":[-120.6700,33.8948,-119.4377,35.1162],"properties":{"kind":"county","name":"Santa Barbara","state":"ca","center":[-120.0539,34.5078],"centroid":[-120.0213,34.7223]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-120.0950,35.1162],[-119.7444,34.9738],[-119.6732,34.9738],[-119.5363,34.8971],[-119.4706,34.9026],[-119.4432,34.9026],[-119.4377,34.4425],[-119.4761,34.3768],[-119.6185,34.4206],[-119.7116,34.3932],[-119.8759,34.4097],[-120.1388,34.4754],[-120.4729,34.4480],[-120.5112,34.5247],[-120.6481,34.5795],[-120.5988,34.7054],[-120.6372,34.7547],[-120.6098,34.8588],[-120.6700,34.9026],[-120.6481,34.9738],[-120.4455,34.9902],[-120.3086,34.9026],[-120.2921,34.9464],[-120.3305,35.0176],[-120.2100,35.0231]]],[[[-119.9142,34.0756],[-119.6404,34.0153],[-119.5911,34.0482],[-119.5199,34.0318],[-119.5582,33.9934],[-119.7937,33.9606],[-119.8759,33.9825]]],[[[-120.3633,34.0756],[-120.3031,34.0263],[-120.4674,34.0372]]],[[[-120.0457,33.9989],[-119.9800,33.9825],[-119.9690,33.9441],[-120.1224,33.8948],[-120.2483,33.9989],[-120.0566,34.0372]]]]}}, 
{"type":"Feature","bbox":[-122.1926,36.9017,-121.2068,37.4822],"properties":{"kind":"county","name":"Santa Clara","state":"ca","center":[-121.6997,37.1925],"centroid":[-121.6986,37.2317]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-121.8531,37.4822],[-121.4752,37.4822],[-121.4587,37.3946],[-121.4094,37.3836],[-121.4040,37.3124],[-121.4587,37.2851],[-121.4040,37.1591],[-121.2835,37.1810],[-121.2287,37.1372],[-121.2451,37.0879],[-121.2068,37.0605],[-121.2506,37.0331],[-121.2451,36.9838],[-121.2177,36.9619],[-121.4204,36.9619],[-121.4478,36.9893],[-121.5792,36.9017],[-121.7381,36.9893],[-121.7161,37.0057],[-121.7545,37.0496],[-122.0283,37.1646],[-122.1543,37.2851],[-122.1926,37.3179],[-122.1926,37.4329],[-122.0831,37.4768],[-121.9243,37.4548]]]]}},   
{"type":"Feature","bbox":[-122.3186,36.8524,-121.5792,37.2851],"properties":{"kind":"county","name":"Santa Cruz","state":"ca","center":[-121.9489,37.0690],"centroid":[-121.9948,37.0536]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-122.1543,37.2851],[-122.0283,37.1646],[-121.7545,37.0496],[-121.7161,37.0057],[-121.7381,36.9893],[-121.5792,36.9017],[-121.6449,36.8962],[-121.6997,36.9181],[-121.8093,36.8524],[-121.9078,36.9674],[-122.1050,36.9564],[-122.2912,37.1098],[-122.3186,37.1865],[-122.1543,37.2139]]]]}}, 
{"type":"Feature","bbox":[-122.4062,38.0299,-121.5902,38.5393],"properties":{"kind":"county","name":"Solano","state":"ca","center":[-121.9982,38.2851],"centroid":[-121.9368,38.2685]},"geometry":{"type":"MultiPolygon","coordinates":[[[[-121.7381,38.5393],[-121.6942,38.5283],[-121.6942,38.3147],[-121.5902,38.3147],[-121.6121,38.1997],[-121.6833,38.1614],[-121.7107,38.0847],[-121.8640,38.0683],[-122.1488,38.0299],[-122.2693,38.0628],[-122.4062,38.1504],[-122.4062,38.1559],[-122.1981,38.1559],[-122.2036,38.3147],[-122.0612,38.3257],[-122.1269,38.4243],[-122.1050,38.5119],[-122.0119,38.4900],[-121.9407,38.5338]]]]}}]
  

def getcounties():

  # r = requests.get("http://25.59.64.230/repo/ca-counties.json")

  # jjc = json.loads(r.text)
  # print len(jjc['features'])
  # counties = {}
  # for f in jjc['features']:
  #   # print f['properties']['name'],f['bbox']
  #   counties[f['properties']['name']] = f['geometry']['coordinates'][0][0]

  # jjc = json.loads(jj)
  counties = {}
  for f in jjcounties:
    # print f['properties']['name'],f['bbox']
    counties[f['properties']['name']] = f['geometry']['coordinates'][0][0]
  return counties


def getcountycoords(county = "San Mateo"):
  for f in jjcounties:
    # print f['properties']['name'],f['bbox']
    if county==f['properties']['name']: 
      return f['geometry']['coordinates'][0][0]
  return []



def getcountiesbbox(county = "San Mateo"):

  # r = requests.get("http://25.59.64.230/repo/ca-counties.json")

  # jjc = json.loads(r.text)
  # print len(jjc['features'])
  # counties = {}
  # for f in jjc['features']:
  #   # print f['properties']['name'],f['bbox']
  #   counties[f['properties']['name']] = f['geometry']['coordinates'][0][0]

  # jjc = json.loads(jj)
  bbox = []
  for f in jjcounties:
    # print f['properties']['name'],f['bbox']
    if county==f['properties']['name']: 
      bbox= f['bbox']
  return bbox




def whichcounty(lon,lat, state='ca'):
  # ts = oc.tnow()

  counties = oc.getcounties()
  cc=""
  for f in counties:
    if oc.pointinpoly(lon,lat, counties[f]):
      # print f
      cc=f
  # print "whichcounty", "%0.3f"%(oc.tnow()-ts)   

  return cc   






def checknear(lon,lat, state ='ca', r=0.004, county="Santa Clara"):
  ccsc = oc.getcountycoords(county = county)
  a=[[lon,lat],[lon+r,lat+r],[lon+r,lat-r],[lon-r,lat+r],[lon-r,lat-r]]
  c=0
  for x in a:
 
    if pointinpoly(x[0], x[1], ccsc):
      c+=1
  if county == "":
    dd = whichcounty(lon,lat)
    if dd!="":
      print lon, lat, "in",  dd
      cc+=4
  return c    

def qkarea(qk):
  lon, lat = oc.qk2lonlat(qk)
  lon10, lat10 = oc.qk2lonlat(oc.qk2qk(qk,1,0))
  lon01, lat01 = oc.qk2lonlat(oc.qk2qk(qk,0,1))

  distx = oc.calcdistlonlat((lon, lat), (lon10, lat10))
  disty = oc.calcdistlonlat((lon, lat), (lon01, lat01))


  print distx, disty, distx* disty
  return  distx* disty


 













