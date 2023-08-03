import os
import sys
import time
import subprocess
from datetime import datetime
import shutil

def cp_tagsets():
    packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl", "psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml", "rsa", "s3transfer", "urllib3", "setuptools", "typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib"]
    packages_l_0 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    packages_l_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
    packages_l_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "plotly", "pycaret", "mahotas", "statsmodels"]
    packages_l.extend(packages_l_0)
    packages_l.extend(packages_l_1)
    packages_l.extend(packages_l_2)
    from itertools import combinations
    # for length in range(1, len(packages_l)+1):
    for length in range(1, 2):  # choose `length` amount of packages
        for package_names in combinations(packages_l, length):
            dirname = os.path.dirname(__file__)
            # out_dirname = dirname
            out_dirname = dirname+"/data/"+"-".join(package_names)+'-'+"tagsets/"
            # print(out_dirname)
            tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
            # print(tagsets_l)
            # if len(tagsets_l) == 2:

            for tagsets_name in tagsets_l[:25]:
                # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/')
                shutil.copy2(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_train/')
            for tagsets_name in tagsets_l[25:33]:
                shutil.copy2(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_SL_biased_test/')

            # # for tagsets_name in tagsets_l[:25]:
            # #     # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/')
            # #     shutil.copy2(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_train/')
            # for tagsets_name in tagsets_l[:2]:
            #     shutil.copy2(out_dirname+tagsets_name, '/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/')

def rm_tagsets():
    from itertools import product
    # packages_l1 = ["opacus", "mahotas"]
    packages_l1 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    packages_l1_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
    packages_l1_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret", "mahotas", "statsmodels"]
    packages_l1.extend(packages_l1_1)
    packages_l1.extend(packages_l1_2)
    packages_l2 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    packages_l2_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
    packages_l2_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret", "mahotas", "statsmodels"]
    packages_l2.extend(packages_l2_1)
    packages_l2.extend(packages_l2_2)
    for package_names in product(packages_l1, packages_l2):
        dirname = os.path.dirname(__file__)
        # out_dirname = dirname
        out_dirname = '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/'
        # print(out_dirname)
        tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
        # print(tagsets_l)
        # if len(tagsets_l) == 2:
        tagset_pattern = "-".join(package_names)
        for tagsets_name in tagsets_l:
            # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/')
            if tagset_pattern in tagsets_name:
                print(out_dirname+tagsets_name)
                if os.path.isfile(out_dirname+tagsets_name):
                    os.remove(out_dirname+tagsets_name)
                else:
                    # If it fails, inform the user.
                    print("Error: %s file not found" % out_dirname+tagsets_name)
    for package_names in product(packages_l2, packages_l1):
        dirname = os.path.dirname(__file__)
        # out_dirname = dirname
        out_dirname = '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/'
        # print(out_dirname)
        tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
        # print(tagsets_l)
        # if len(tagsets_l) == 2:
        tagset_pattern = "-".join(package_names)
        for tagsets_name in tagsets_l:
            # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/')
            if tagset_pattern in tagsets_name:
                print(out_dirname+tagsets_name)
                if os.path.isfile(out_dirname+tagsets_name):
                    os.remove(out_dirname+tagsets_name)
                else:
                    # If it fails, inform the user.
                    print("Error: %s file not found" % out_dirname+tagsets_name)


if __name__ == '__main__':
    cp_tagsets()