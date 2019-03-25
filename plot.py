# MIT License
# 
# Copyright (c) 2019 Miikka Väisälä
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import pylab as plt 
import numpy as np

def read_bin(fname, fdir, fnum, numtype=np.float32):
    '''Read in a floating point array'''
    filename = fdir + fname + '_' + str(fnum) + '.map'
    datas = np.DataSource()
    read_ok = datas.exists(filename)
    if read_ok:
        print(filename)
        array = np.fromfile(filename, dtype=numtype)

        angle = array[0]

        NX = np.int(np.sqrt(array[1:].size))

        print("size", array[1:].size, NX)

        array = np.reshape(array[1:], (NX, NX), order='F')
    else:
        array = None
        NX = None
     
    return array, NX, read_ok

savename = "happy_face"

for fnum in range(360): 
    colden_map, NX, read_ok = read_bin("happy_face", "", fnum)

    if read_ok:
        print(colden_map.dtype)
        #colden_map = np.array(colden_map, dtype = np.float32)
  
        print(colden_map)
        print(colden_map.dtype)
        print(np.amax(colden_map))
        print(np.shape(colden_map))

        fig = plt.figure()
        plt.imshow(colden_map, cmap=plt.get_cmap('inferno'), origin='lower', vmin=0.0, vmax=3000.0)

        fnum = str(fnum)
        framenum = fnum.zfill(4)

        plt.savefig('%s_%s.png' % (savename, framenum))
        print('Saved %s_%s.png' % (savename, framenum))
        plt.close(fig)

