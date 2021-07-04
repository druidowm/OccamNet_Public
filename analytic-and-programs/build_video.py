import io
import os
import base64
import shutil

fps = 10
video_folder = './'
temp_folder = './video_temp/'

path = video_folder + 'generated.mp4'
if os.path.exists(path):
    os.remove(path)

ffmpeg_op = 'ffmpeg -r ' + str(fps)
ffmpeg_op += " -i " + temp_folder + "/temp_%01d.png"
ffmpeg_op += " -vcodec libx264 -crf 25 -pix_fmt yuv420p "
ffmpeg_op += path

os.system(ffmpeg_op)
