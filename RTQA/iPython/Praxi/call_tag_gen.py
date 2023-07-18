import os
import sys
import time
import subprocess
from datetime import datetime


def praxi():
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl", "psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml", "rsa", "s3transfer", "urllib3", "setuptools", "typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "emoji", "tdqm", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib"]
    # packages_l_0 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    # packages_l_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
    # packages_l_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret", "mahotas", "statsmodels"]
    # packages_l.extend(packages_l_0)
    # packages_l.extend(packages_l_1)
    # packages_l.extend(packages_l_2)

    packages_l = ["opacus"]
    from itertools import combinations
    # for length in range(1, len(packages_l)+1):
    for length in range(1, 2):
        for package_names in combinations(packages_l, length):
            for _ in range(1):
                # packages_str = " ".join(package_names)
                # cmd1 = "/home/cc/Praxi-study/data_gen_venv/venv/bin/python3 -m pip install "+packages_str
                # cmd2 = "/home/cc/Praxi-study/data_gen_venv/venv/bin/python3 -m pip uninstall -y " +packages_str
                # print(cmd1)
                # print(cmd2)
                
                # # p = subprocess.Popen(["which", "pip"], stdin=subprocess.PIPE)
                # # p.communicate()

                # # install dependencies (not recorded)
                # p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE)
                # p_cmd1.communicate()

                # # delete existing package
                # p_cmd2 = subprocess.Popen(cmd2.split(" "), stdin=subprocess.PIPE)
                # p_cmd2.communicate()


                dirname = os.path.dirname(__file__)
                # out_dirname = dirname
                out_dirname = dirname+"/data"
                # # Start Deltashocker filesystem change recorder
                # # p = subprocess.Popen(['python3', os.path.join(dirname, 'cs_rec.py'),'-t',os.path.join(dirname, 'changesets'),'-l',time_string], stdin=subprocess.PIPE)
                # # p = subprocess.Popen(['python3', os.path.join(dirname, 'cs_rec.py'),'-t',os.path.join(dirname, 'changesets'),'-l', *package_names], stdin=subprocess.PIPE)
                # p = subprocess.Popen(['/home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/.venv/bin/python3', os.path.join(dirname, 'cs_rec.py'),'-t',os.path.join(out_dirname),'-l', *package_names], stdin=subprocess.PIPE)

                # # install packages
                # p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE)
                # p_cmd1.communicate()

                # p.communicate(input=b'\n')

                # # clean up
                # p_cmd2 = subprocess.Popen(cmd2.split(" "), stdin=subprocess.PIPE)
                # p_cmd2.communicate()
                
                # tagset generator
                labels_str = "-".join(package_names)
                print(os.path.join(out_dirname, labels_str+'-changesets'))
                p2 = subprocess.Popen(['/home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/.venv/bin/python3', os.path.join(dirname, 'tagset_gen.py'),'-c',os.path.join(out_dirname, labels_str+'-changesets'),'-t',os.path.join(out_dirname, labels_str+'-tagsets')], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                # main.py generates a trains a new model everytime from scratch from the tagsets it is given
                # demo_tagsets/<tagset_directory> is what vw is testing against the sample
                out, err = p2.communicate()
                print(out)
                print(err)

                # # predictions generator
                # p3 = subprocess.Popen(['python3', os.path.join(dirname, 'main.py'),'-t',os.path.join(dirname, 'demo_tagsets/sl_train_tag'),
                #     '-s',os.path.join(dirname, 'tagsets'),'-o',os.path.join(dirname, 'results'), '-i', os.path.join(dirname, 'iter_model.vw'), '-l'], 
                #     stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                # (output, err) = p3.communicate()
                # print(output, err)
                # print("!!!!!!!!!!!! p3 Curl start!!!!!")
                # # curl --header "Content-Type: application/json" --data "{'test':'test'}" --request POST http://10.106.160.36:6025/train/
                # p3 = subprocess.Popen(['curl', "--header","Content-Type: application/json",'--data',"{'test':'test'}", "--request", "POST", "http://10.106.160.36:6025/train/"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                # # p3 = subprocess.Popen(['python3', os.path.join(dirname, 'main.py'),'-t',os.path.join(dirname, 'tagsets'),
                # #     '-s',os.path.join(dirname, 'tagsets'),'-o',os.path.join(dirname, 'results'), '-i', os.path.join(dirname, 'iter_model.vw'), '-l'], 
                # #     stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                # # p3 = subprocess.Popen(['python3', os.path.join(dirname, 'main.py'),'-t',os.path.join(dirname, 'demo_tagsets/iter_init'),
                # #     '-s',os.path.join(dirname, 'demo_tagsets/sl_test_tag'),'-o',os.path.join(dirname, 'results'), '-i', os.path.join(dirname, 'iter_model.vw'), '-l'], 
                # #     stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                # (output, err) = p3.communicate()
                # print(output, err)
                # print("!!!!!!!!!!!!")
    return

if __name__ == '__main__':
    praxi()