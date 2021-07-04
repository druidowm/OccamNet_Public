import io
import os
import base64
import shutil

from IPython.display import HTML

class VideoSaver:
    def __init__(self, temp_folder='./video_temp',
                video_folder='./experiments/videos/', video_name='video'):

        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder)

        if not os.path.isdir(video_folder):
            os.mkdir(video_folder)

        self.temp_folder = temp_folder
        self.video_folder = video_folder
        self.video_name = video_name + '.mp4'
        self.frame = 0
        self.closed = False

    def snap(self, plot):
        if self.closed: raise Exception("Saver was Closed")
        if self.frame == 0:
            os.mkdir(self.temp_folder)

        plot.savefig(self.temp_folder + '/temp_' + str(self.frame) + '.png')
        # print("Saved a snap at " + self.temp_folder + '/temp_' + str(self.frame) + '.png')
        self.frame += 1
        plot.close()

    def save(self, fps=10):
        if self.closed: raise Exception("Saver was Closed")
        path = self.video_folder + self.video_name
        if os.path.exists(path):
            os.remove(path)

        ffmpeg_op = 'ffmpeg -r ' + str(fps)
        ffmpeg_op += " -i " + self.temp_folder + "/temp_%01d.png"
        ffmpeg_op += " -vcodec libx264 -crf 25 -pix_fmt yuv420p "
        ffmpeg_op += path

        os.system(ffmpeg_op)

    def render(self):
        video = io.open(self.video_folder + self.video_name, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''
            <video width="400" height="auto" alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii')))

    def close(self):
        self.frame = 0
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        self.closed = True
