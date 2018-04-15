import os, sys, json


 wflist = [[380,440],[650,720],[1080,1210],[1380,1470],[1780,1930],[2130,2230]]
 
#wflist = [[380,440,]]

date = "20171130"
tiledir = "/home/oc/sdb1/processed/20171118test19/"

zlist = [[17,6],[19,18],[21,20]]
fn ="1"


wfarr = []
mfarr = []
for wfdo in wflist:
  for n in range(wfdo[0], wfdo[1]):
    iy = int(n/999) 
    ix = n%999
    imuri = "/"+date+"/wf/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
    wfarr.append(imuri)
    for mfn in range(7):
       mf = "mf%02d"%mfn
       imuri = "/"+date+"/"+mf +"/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
       jj = {"imuri": imuri}
       mfarr.append(imuri)


wffn = "wfsall.txt"
with open(wffn, 'w') as f: f.write('')

for wf in wfarr: 
  js = {"imuri": wf}
  with open(wffn, 'a') as f: f.write(json.dumps(js)+'\n')

mffn = "mfsall.txt"
with open(mffn, 'w') as f: f.write('')

for mf in mfarr:
  js = {"imuri": mf}
  with open(mffn, 'a') as f: f.write(json.dumps(js)+'\n')


outmflist = "mflistall"+date+".txt"
with open(outmflist, 'w') as f: f.write('')
for wfdo in wflist:
  for n in range(wfdo[0], wfdo[1]):
     iy = int(n/999) 
     ix = n%999
     for mfn in range(7):
       mf = "mf%02d"%mfn
       imuri = "/"+date+"/"+mf +"/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
       jj = {"imuri": imuri}
       with open(outmflist, 'a') as f: f.write(json.dumps(jj)+'\n')

print "wrote", outmflist



wfj=""

tc =0
for wfdo in wflist:  
  tc += wfdo[1]-wfdo[0] 
  tmpfn = "/tmp/qklistwf-"+fn+".txt"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+ " 10 \n"
  wfj+= "./wfest -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -mixdir  "+tiledir+"/  \n"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+" \n"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+ " 10 \n"
  wfj+= "./gcutwf -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -vedir  /home/oc/sdb1/ve/ -step 1 -mixdir  "+tiledir+"/    -outfn "+tmpfn+" -imgsrc /home/oc/sdb1/ -zout 19 \n"
  wfj+= "./tileshrink -infn  "+tmpfn+"    -mixdir "+tiledir+"  -minzoom 18 \n"
  wfj+= "./gcutwf -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -vedir  /home/oc/sdb1/ve/ -step 1 -mixdir  "+tiledir+"/  -outfn "+tmpfn+"  -imgsrc /home/oc/sdb1/ -zout 17 \n"
  wfj+= "./tileshrink -infn  "+tmpfn+"    -mixdir "+tiledir+"  -minzoom 6 \n"
 
with open("wfj.txt", "w") as fout: fout.write(wfj)
 
print tc
