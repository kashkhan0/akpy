## instructions part1


```

# install ubuntu 17.04
# install opencv https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh

# go to github repo
#  https://github.com/kashkhan0/akpy/tree/r2

# clone the repository

git clone https://github.com/kashkhan0/akpy.git

# check out the branch

git checkout r2

# use miniconda or venv
pip install requests
pip install arrow
pip install pillow

# put required lon lat coordinates in lonlat.txt

# download the tiles 

python vepull.py


# copy your image files to a directory 

find ./images -name "*.JPG" > imglist.txt

# feed the image list to fdbfiles to extract exif

python fdbfile.py imglist.txt

# copy the gps track to text file of the format of aocgs20171117.txt

# find centers using  ocgsave20171117.py

python ocgsave20171117.py

# this will generate 20171117centersingle.txt

# continued in part II

# generate downsampled images using makesmall.py 

# when reach tjos stage contact me

```