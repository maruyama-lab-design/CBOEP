import gzip
import shutil
import glob

files = glob.glob("D:/ylwrv/TransEPI-main/data/BENGI/original/*.gz")
for source_file in files:
	target_file = source_file[:-3]
	with gzip.open(source_file, mode="rb") as gzip_file:
		with open(target_file, mode="wb") as decompressed_file:
			shutil.copyfileobj(gzip_file, decompressed_file)