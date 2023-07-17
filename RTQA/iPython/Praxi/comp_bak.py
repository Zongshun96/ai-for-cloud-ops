import os

def dir_content_compare():
    out_dirname = "/home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/data"
    tagsets_0_l = set([name[:-11] for name in os.listdir(out_dirname) if name[-11:] == '-changesets'])
    out_dirname = "/home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/data_bak"
    tagsets_1_l = set([name[:11] for name in os.listdir(out_dirname)])

    packages_l = set(["MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl", "psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml", "rsa", "s3transfer", "urllib3", "setuptools", "typing-extensions", "charset-normalizer", "certi", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "emoji", "tdqm", "Flask", "seaborn", "NLTK", "pytest", "zipp", "prorobuf", "authlib", "docuils", "grpcio", "pycparser", "colorama", "oauthlib"])


    print(tagsets_0_l.difference(tagsets_1_l))
    print(tagsets_1_l.difference(tagsets_0_l))

if __name__ == '__main__':
    dir_content_compare()