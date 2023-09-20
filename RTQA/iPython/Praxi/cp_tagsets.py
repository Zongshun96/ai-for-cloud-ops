import os
import sys
import time
import subprocess
from datetime import datetime
import shutil
from pathlib import Path

def cp_tagsets():
    # ============= data_0
    packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl"]
    packages_l0 = ["psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml"]
    packages_l1 = ["typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata"]
    packages_l2 = ["emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib", "pandas"]
    packages_l3 = ["cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn"]
    packages_l4 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap"]
    packages_l5 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "plotly", "pycaret", "mahotas", "statsmodels", "nilearn", "networkx"]
    packages_l6 = ["SQLAlchemy", "matplotlib", "scipy", "boto3", "rsa", "s3transfer", "urllib3", "setuptools", "pyspark", "pillow"]
    packages_l.extend(packages_l0)
    packages_l.extend(packages_l1)
    packages_l.extend(packages_l2)
    packages_l.extend(packages_l3)
    packages_l.extend(packages_l4)
    packages_l.extend(packages_l5)
    packages_l.extend(packages_l6)
    # ============= data_1
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi"]
    # packages_l_3 = ["azure-core", "greenlet", "filelock", "pluggy", "paramiko"]
    # packages_l.extend(packages_l_3)
    # ============= data_2
    # packages_l = ["numpy", "tzdata", "contourpy", "cycler", "fonttools", "kiwisolver", "packaging", "pyparsing", "importlib-resources"]
    from itertools import combinations
    # for length in range(1, len(packages_l)+1):
    for length in range(1, 2):  # choose `length` amount of packages
        for package_names in combinations(packages_l, length):
            dirname = os.path.dirname(__file__)
            # out_dirname = dirname
            out_dirname = dirname+"/data_0/"+"-".join(package_names)+'-'+"tagsets/"
            # print(out_dirname)
            if not os.path.isdir(out_dirname):
                continue
            tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
            # print(tagsets_l)
            # if len(tagsets_l) == 2:

            for tagsets_name in tagsets_l[:25]:
                # big_ML_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/'
                # Path(big_ML_biased_test).mkdir(parents=True, exist_ok=True)
                # print(out_dirname+tagsets_name, big_ML_biased_test)
                big_train = '/home/cc/Praxi-study/Praxi-Pipeline/data/data_0_/big_train/'
                Path(big_train).mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_dirname+tagsets_name, big_train)
            # for tagsets_name in tagsets_l[25:33]:
            #     big_SL_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/big_SL_biased_test/'
            #     Path(big_SL_biased_test).mkdir(parents=True, exist_ok=True)
            #     shutil.copy2(out_dirname+tagsets_name, big_SL_biased_test)

            # for tagsets_name in tagsets_l[:2]:
            #     big_ML_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/data_0_/big_ML_biased_test/'
            #     Path(big_ML_biased_test).mkdir(parents=True, exist_ok=True)
            #     shutil.copy2(out_dirname+tagsets_name, big_ML_biased_test)

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