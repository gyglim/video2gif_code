from setuptools import setup

setup(name='video2gif_code',
      version='0.1',
      description='This shows how to use or pretrained model of video2gif to score segments',
      url='https://github.com/gyglim',
      author='Michael Gygli',
      license='BSD',
      packages=['video2gif'],
      install_requires=[
          'numpy','moviepy','theano','lasagne','scikit-image','tqdm','pandas'],
      zip_safe=False)