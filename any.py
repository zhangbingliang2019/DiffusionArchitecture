# import os
# file = ["1066674784.mp4",
# "1066674790.mp4",
# "1066674802.mp4",
# "1066674877.mp4",
# "1066674883.mp4",
# "1066674916.mp4",
# "1066674928.mp4",
# "1066674931.mp4",
# "1066674934.mp4",
# "1066674937.mp4",
# "1066674940.mp4",
# "1066674943.mp4",
# "1066674946.mp4",
# "1066674949.mp4",
# "1066674952.mp4",
# "1066674958.mp4"]
#
# command = ("scp -r -P 61201 bingliang@rp-a100-"
#            "80gb-8x-hba001.cloud.together.ai:"
#            "/home/bingliang/data/WebVid2.5M/videos/000001_000050/{} videos")
#
# for p in file:
#     print(p)
#     os.system(command.format(p))

import pylab
import imageio
filename = 'videos/1066674784.mp4'
import skvideo.io
videodata = skvideo.io.vread(filename)
print(videodata.shape)