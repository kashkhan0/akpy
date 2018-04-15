
from time import sleep
import sys, os, random, datetime, json
 
import string, re
from decimal import Decimal
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
from PIL.ImageOps import autocontrast
 
import subprocess 
import json 
import arrow
import numpy as np
import shutil
from oc import Config 
 
 
 
def tnow( text = "tt"):
  e = arrow.utcnow()
  return  e.timestamp + e.microsecond / 1e6
 
 
 
 
def resize(inpath,outdir, outfn, adj=0):
      if os.path.isfile(outfn):
        print "exists", outfn
        print "\033[1;37;44m inliers " ,1, "/",6, " \033[0m"
      #  return
      res = os.system('mkdir -p '+ outdir)
      try:
        im = Image.open(inpath)
      except:
        print "bad img",inpath
        return
 
      (width, height) = im.size
      swidth, sheight = width*0.16,height*0.16
      try:  
        im.thumbnail( (swidth, sheight ), Image.ANTIALIAS)
      except:
        return    
      if adj > 0 :
        imtmp = autocontrast(im,0.25)
        im = imtmp
      im.save(outfn)
 
   
def test():
 
  fullimdir = "/home/oc/sdb1/"
  procimdir = "/home/oc/sdb1/processed/"
   
  imuri = "/20171118/sw/112D3200/DSC_0030.JPG"
  inpath = fullimdir + imuri
  outdir =  procimdir + imuri +'/'
  outfn = outdir+"small.jpg"
  resize(inpath,outdir, outfn,2)
 
 
if __name__ == "__main__":
 
  avgt = 0.1
  fullimdir = "/home/oc/sdb1/"
  procimdir = "/home/oc/sdb1/processed/"
  #  test()
 
  flist = []
  infn = "dont.txt"
  if len(sys.argv) > 1:
    infn = sys.argv[1]
 
  if not os.path.isfile(infn):
    print "no", infn
    quit()
   
 
  with open(infn) as fh:
    for line in fh:
      j = json.loads(line)
       
      try: 
        imuri = j["imuri"]
 
      except:
        print "bad", line
        continue
 
       
    
      ts= tnow()
      inpath = fullimdir + imuri
      outdir =  procimdir + imuri +'/'
      outfn = outdir+"small.jpg"
      if os.path.isfile(outfn): continue
      resize(inpath,outdir, outfn,2)
      
      te=tnow()
      avgt = 0.9*avgt + 0.1 * (te-ts)
      print 'make ', outfn, "%0.3f"%(avgt)