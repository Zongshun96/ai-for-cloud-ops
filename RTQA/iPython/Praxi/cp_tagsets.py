import os
import sys
import time
import subprocess
from datetime import datetime
import shutil
from pathlib import Path

def cp_tagsets():
    # # ============= data_0
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl"]
    # packages_l0 = ["psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml"]
    # packages_l1 = ["typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata"]
    # packages_l2 = ["emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib", "pandas"]
    # packages_l3 = ["cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn"]
    # packages_l4 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap"]
    # packages_l5 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "plotly", "pycaret", "mahotas", "statsmodels", "nilearn", "networkx"]
    # packages_l6 = ["SQLAlchemy", "matplotlib", "scipy", "boto3", "rsa", "s3transfer", "urllib3", "setuptools", "pyspark", "pillow"]
    # packages_l.extend(packages_l0)
    # packages_l.extend(packages_l1)
    # packages_l.extend(packages_l2)
    # packages_l.extend(packages_l3)
    # packages_l.extend(packages_l4)
    # packages_l.extend(packages_l5)
    # packages_l.extend(packages_l6)
    # ============= data_1
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi"]
    # packages_l_3 = ["azure-core", "greenlet", "filelock", "pluggy", "paramiko"]
    # packages_l.extend(packages_l_3)
    # ============= data_2
    # packages_l = ["numpy", "tzdata", "contourpy", "cycler", "fonttools", "kiwisolver", "packaging", "pyparsing", "importlib-resources"]
    # ============= data_3
    packages_l_4 = ['certifi', 'numpy', 'packaging', 'aiobotocore', 'protobuf', 'jmespath', 'googleapis-common-protos', 'platformdirs', 'google-auth', 'werkzeug', 'pydantic', 'filelock', 'pyparsing', 'async-timeout', 'aiohttp', 'docutils', 'pyarrow', 'exceptiongroup', 'pluggy', 'lxml', 'requests-oauthlib', 'tqdm', 'pyasn1-modules', 'azure-core', 'decorator', 'pyopenssl', 'greenlet', 'importlib-resources', 'multidict', 'pygments', 'websocket-client', 'pymysql', 'distlib', 'coverage', 'aiosignal', 'et-xmlfile', 'openpyxl', 'chardet', 'google-cloud-core', 'google-cloud-storage', 'asn1crypto', 'tabulate', 'google-api-python-client', 'referencing', 'iniconfig', 'tomlkit', 'rpds-py', 'paramiko', 'gitpython', 'jsonschema-specifications', 'requests-toolbelt', 'pynacl', 'more-itertools', 'proto-plus', 'psycopg2-binary', 'itsdangerous', 'azure-storage-blob', 'msal', 'google-resumable-media', 'bcrypt', 'pathspec', 'tzlocal', 'anyio', 'grpcio-tools', 'google-cloud-bigquery', 'docker', 'cython', 'mdit-py-plugins', 'joblib', 'regex', 'mypy-extensions', 'smmap', 'gitdb', 'sagemaker', 'sqlparse', 'msgpack', 'wcwidth', 'google-auth-oauthlib', 'poetry-core', 'sniffio', 'py', 'pycryptodomex', 'pyrsistent', 'azure-common', 'future', 'dnspython', 'pexpect', 'ptyprocess', 'msrest', 'jaraco-classes', 'dill', 'portalocker', 'ruamel-yaml', 'markdown', 'snowflake-connector-python', 'py4j', 'tornado', 'keyring', 'google-crc32c', 'prompt-toolkit', 'markdown-it-py', 'tenacity', 'cloudpickle', 'httplib2', 'rich', 'alembic', 'gunicorn', 'tzdata', 'awswrangler', 'fonttools', 'azure-identity', 'threadpoolctl', 'msal-extensions', 'xmltodict', 'kiwisolver', 'pycodestyle', 'termcolor', 'python-dotenv', 'tb-nightly', 'scramp', 'backoff', 'uritemplate', 'toml', 'jedi', 'webencodings', 'cachecontrol', 'marshmallow', 'poetry-plugin-export', 'ipython', 'h11', 'mccabe', 'nest-asyncio', 'cycler', 'ply', 'sortedcontainers', 'pycryptodome', 'pg8000', 'google-auth-httplib2', 'trove-classifiers', 'oscrypto', 'traitlets', 'mako', 'pyodbc', 'pkgutil-resolve-name', 'pyzmq', 'prometheus-client', 'redshift-connector', 'isort', 'toolz', 'jeepney', 'httpcore', 'secretstorage', 'adal', 'pytest-cov', 'shellingham', 'babel', 'blinker', 'datadog', 'typing-inspect', 'black', 'pymongo', 'jsonpointer', 'jupyter-client', 'defusedxml', 'google-cloud-pubsub', 'argcomplete', 'httpx', 'tensorboard', 'pyflakes', 'jupyter-core', 'sentry-sdk', 'xlrd', 'flake8', 'poetry', 'cfn-lint', 'pkginfo', 'fastapi', 'nbconvert', 'mdurl', 'requests-aws4auth', 'parso', 'asynctest', 'contourpy', 'pydantic-core', 'python-json-logger', 'absl-py', 'jsonpath-ng', 'databricks-cli', 'python-utils', 'google-cloud-bigquery-storage', 'nbformat', 'pickleshare', 'backcall', 'fastjsonschema', 'notebook', 'progressbar2', 'astroid', 'aioitertools', 'mistune', 'starlette', 'rapidfuzz', 'matplotlib-inline', 'opensearch-py', 'appdirs', 'lazy-object-proxy', 'jupyter-server', 'tensorflow', 'ipykernel', 'pbr', 'pylint', 'transformers', 'arrow', 'h5py', 'kubernetes', 'build', 'jsonpatch', 'imageio', 'setuptools-scm', 'bleach', 'huggingface-hub', 'asgiref', 'annotated-types', 'websockets', 'html5lib', 'debugpy', 'cattrs', 'pyproject-hooks', 'entrypoints', 'grpc-google-iam-v1', 'uvicorn', 'mlflow', 'smart-open', 'oauth2client', 'altair', 'msrestazure', 'multiprocess', 'numba', 'tinycss2', 'dulwich', 'llvmlite', 'tensorflow-estimator', 'zope-interface', 'lockfile', 'elasticsearch', 'mock', 'google-pasta', 'flatbuffers', 'retry', 'aiofiles', 'google-cloud-secret-manager', 'pygithub', 'mypy', 'humanfriendly', 'requests-file', 'shapely', 'orjson', 'crashtest', 'great-expectations', 'aenum', 'pysocks', 'cleo', 'comm', 'httptools', 'gast', 'querystring-parser', 'nodeenv', 'nbclient', 'tensorboard-data-server', 'contextlib2', 'identify', 'xlsxwriter', 'cached-property', 'azure-storage-file-datalake', 'croniter', 'tox', 'deepdiff', 'tokenizers', 'django', 'notebook-shim', 'send2trash', 'mysql-connector-python', 'ipywidgets', 'configparser', 'pendulum', 'execnet', 'jupyterlab-server', 'widgetsnbextension', 'text-unidecode', 'rfc3339-validator', 'overrides', 'pre-commit', 'typer', 'keras', 'json5', 'semver', 'watchdog', 'hvac', 'responses', 'torch', 'jupyterlab', 'pytzdata', 'aws-sam-translator', 'snowflake-sqlalchemy', 'python-slugify', 'cfgv', 'pipenv', 'asttokens', 'argon2-cffi', 'installer', 'dataclasses', 'sphinx', 'jupyterlab-widgets', 'executing', 'gremlinpython', 'distro', 'typeguard', 'azure-mgmt-core', 'selenium', 'jupyter-events', 'pytest-xdist', 'click-plugins', 'stack-data', 'pytest-mock', 'azure-storage-common', 'confluent-kafka', 'slack-sdk', 'pure-eval', 'opt-einsum', 'rfc3986', 'xgboost', 'tblib', 'dataclasses-json', 'opentelemetry-sdk', 'apache-airflow', 'uri-template', 'fastavro', 'tensorflow-serving-api', 'ipython-genutils', 'sentencepiece', 'futures', 'tensorflow-io-gcs-filesystem', 'sympy', 'unidecode', 'xxhash', 'safetensors', 'db-dtypes', 'pandocfilters', 'prettytable', 'patsy', 'opentelemetry-api', 'retrying', 'docopt', 'azure-mgmt-resource', 'mpmath', 'gcsfs', 'async-lru', 'jupyterlab-pygments', 'astunparse', 'setproctitle', 'terminado', 'libclang', 'pytest-runner', 'thrift', 'jsonpickle', 'semantic-version', 'ordered-set', 'azure-keyvault-secrets', 'pymssql', 'faker', 'pysftp', 'webcolors', 'argon2-cffi-bindings', 'jupyter-lsp', 'typing', 'rfc3986-validator', 'zeep', 'inflection', 'antlr4-python3-runtime', 'sphinxcontrib-serializinghtml', 'azure-datalake-store', 'graphviz', 'boto', 'fqdn', 'isoduration', 'jupyter-server-terminals', 'deprecation', 'moto', 'snowballstemmer', 'openai', 'opentelemetry-proto', 'distributed', 'azure-graphrbac', 'typed-ast', 'sphinxcontrib-htmlhelp', 'sphinxcontrib-applehelp', 'sphinxcontrib-devhelp', 'sphinxcontrib-qthelp', 'opencensus', 'ujson', 'opencensus-context', 'aioconsole', 'pathos', 'libcst', 'parsedatetime', 'stevedore', 'python-gnupg', 'google-cloud-firestore', 'pyproj', 'pandas-gbq', 'pox', 'trio', 'ppft', 'gspread', 'applicationinsights', 'numexpr', 'gevent', 'zope-event', 'kombu', 'shap', 'argparse', 'opentelemetry-exporter-otlp-proto-http', 'trio-websocket', 'google-cloud-appengine-logging', 'email-validator', 'structlog', 'loguru', 'watchtower', 'pyathena', 'torchvision', 'azure-mgmt-keyvault', 'azure-mgmt-storage', 'simple-salesforce', 'checkov', 'coloredlogs', 'apache-beam', 'tensorboard-plugin-wit', 'gsutil', 'kafka-python', 'mypy-boto3-rds', 'celery', 'pathlib2', 'pycrypto', 'wandb', 'colorlog', 'enum34', 'pybind11', 'tldextract', 'prometheus-flask-exporter', 'opentelemetry-semantic-conventions', 'types-urllib3', 'azure-cosmos', 'azure-eventhub', 'djangorestframework', 'opencensus-ext-azure', 'docstring-parser', 'lz4', 'pydata-google-auth', 'pywavelets', 'lightgbm', 'datetime', 'ecdsa', 'pyhcl', 'uamqp', 'cligj', 'google-cloud-resource-manager', 'slicer', 'fire', 'makefun', 'python-jose', 'azure-mgmt-containerregistry', 'imagesize', 'google-cloud-logging', 'keras-preprocessing', 'unittest-xml-reporting', 'alabaster', 'flask-cors', 'schema', 'hpack', 'nvidia-cudnn-cu11', 'partd', 'delta-spark', 'nvidia-cublas-cu11', 'wsproto', 'amqp', 'hypothesis', 'pytest-asyncio', 'python-http-client', 'validators', 'h2', 'azure-mgmt-authorization', 'databricks-sql-connector', 'sshtunnel', 'hyperframe', 'spacy', 'unicodecsv', 'brotli', 'fiona', 'locket', 'apache-airflow-providers-common-sql', 'holidays']

    from itertools import combinations
    # for length in range(1, len(packages_l)+1):
    for length in range(2, 3):  # choose `length` amount of packages
        for package_names in combinations(packages_l_4, length):
            dirname = os.path.dirname(__file__)
            # out_dirname = dirname
            out_dirname = dirname+"/data_3/"+"-".join(package_names)+'-'+"tagsets/"
            # print(out_dirname)
            if not os.path.isdir(out_dirname):
                continue
            tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
            # print(tagsets_l)
            # if len(tagsets_l) == 2:

            # for tagsets_name in tagsets_l[:25]:
            #     # big_ML_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/'
            #     # Path(big_ML_biased_test).mkdir(parents=True, exist_ok=True)
            #     # print(out_dirname+tagsets_name, big_ML_biased_test)
            #     big_train = '/home/cc/Praxi-study/Praxi-Pipeline/data/data_3/big_train/'
            #     Path(big_train).mkdir(parents=True, exist_ok=True)
            #     shutil.copy2(out_dirname+tagsets_name, big_train)
            # for tagsets_name in tagsets_l[25:33]:
            #     big_SL_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/big_SL_biased_test/'
            #     Path(big_SL_biased_test).mkdir(parents=True, exist_ok=True)
            #     shutil.copy2(out_dirname+tagsets_name, big_SL_biased_test)

            for tagsets_name in tagsets_l[:2]:
                big_ML_biased_test = '/home/cc/Praxi-study/Praxi-Pipeline/data/data_3/big_ML_biased_test/'
                Path(big_ML_biased_test).mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_dirname+tagsets_name, big_ML_biased_test)

# def rm_tagsets():
#     from itertools import product
#     # packages_l1 = ["opacus", "mahotas"]
#     packages_l1 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
#     packages_l1_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
#     packages_l1_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret", "mahotas", "statsmodels"]
#     packages_l1.extend(packages_l1_1)
#     packages_l1.extend(packages_l1_2)
#     packages_l2 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
#     packages_l2_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
#     packages_l2_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret", "mahotas", "statsmodels"]
#     packages_l2.extend(packages_l2_1)
#     packages_l2.extend(packages_l2_2)
#     for package_names in product(packages_l1, packages_l2):
#         dirname = os.path.dirname(__file__)
#         # out_dirname = dirname
#         out_dirname = '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/'
#         # print(out_dirname)
#         tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
#         # print(tagsets_l)
#         # if len(tagsets_l) == 2:
#         tagset_pattern = "-".join(package_names)
#         for tagsets_name in tagsets_l:
#             # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/')
#             if tagset_pattern in tagsets_name:
#                 print(out_dirname+tagsets_name)
#                 if os.path.isfile(out_dirname+tagsets_name):
#                     os.remove(out_dirname+tagsets_name)
#                 else:
#                     # If it fails, inform the user.
#                     print("Error: %s file not found" % out_dirname+tagsets_name)
#     for package_names in product(packages_l2, packages_l1):
#         dirname = os.path.dirname(__file__)
#         # out_dirname = dirname
#         out_dirname = '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/'
#         # print(out_dirname)
#         tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
#         # print(tagsets_l)
#         # if len(tagsets_l) == 2:
#         tagset_pattern = "-".join(package_names)
#         for tagsets_name in tagsets_l:
#             # print(out_dirname+tagsets_name, '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/')
#             if tagset_pattern in tagsets_name:
#                 print(out_dirname+tagsets_name)
#                 if os.path.isfile(out_dirname+tagsets_name):
#                     os.remove(out_dirname+tagsets_name)
#                 else:
#                     # If it fails, inform the user.
#                     print("Error: %s file not found" % out_dirname+tagsets_name)


if __name__ == '__main__':
    cp_tagsets()