import os
import zipfile

# Extracting zipped files from the source

os.system("wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O tmp/rps.zip")

os.system("!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O tmp/rps-test-set.zip")

local_zip = 'tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = 'tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

# Generating the folders

rock_dir = os.path.join('tmp/rps/rock')
paper_dir = os.path.join('tmp/rps/paper')
scissors_dir = os.path.join('tmp/rps/scissors')

rock_count = len(os.listdir(rock_dir))
paper_count = len(os.listdir(paper_dir))
scissors_count = len(os.listdir(scissors_dir))
# images_count = rock_count + paper_count + scissors_count
print('total training rock images:', rock_count)
print('total training paper images:', paper_count)
print('total training scissors images:', scissors_count)

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
pass
