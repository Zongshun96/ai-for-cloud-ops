import os
import sys
import time
import subprocess
from datetime import datetime
import multiprocessing as mp
from itertools import combinations
from itertools import product
from pathlib import Path
import yaml

def build_logger(logger_name, logdirpath):
    import logging
    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Create handlers
    # print(logdirpath)
    Path(logdirpath).mkdir(parents=True, exist_ok=True)
    f_handler = logging.FileHandler(filename=logdirpath+'file.log')
    c_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler.setLevel(logging.INFO)
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)
    
    return logger

def praxi(packages_l, packages_l_1=None, venv_dir="/home/cc/Praxi-study/data_gen_venv/venv/", length=2):

    if packages_l_1!= None:
        selection_set = product(packages_l, packages_l_1)
    else:
        selection_set = combinations(packages_l, length)
    # for length in range(1, len(packages_l)+1):
    # for length in range(2, 3):
    for package_names in selection_set:
    # for package_names in product(packages_l, packages_l_1):
        labels_str = "-".join(package_names)
        packages_str = " ".join(package_names)
        dirname = os.path.dirname(__file__)
        # # out_dirname = dirname
        out_dirname = dirname+"/data3_par_test"

        logger = build_logger(labels_str, os.path.join(out_dirname, labels_str+'-changesets/'))
        
        cmd0 = "python3 -m venv "+venv_dir
        p_cmd0 = subprocess.Popen(cmd0.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p_cmd0.communicate()
        # print(output)
        # print(error)
        logger.debug(output)
        logger.debug(error)

        
        for _ in range(2):
            cmd1 = venv_dir+"bin/python3 -m pip install "+packages_str
            cmd2 = venv_dir+"bin/python3 -m pip uninstall -y " +packages_str
            # print(cmd1)
            # print(cmd2)
            logger.info(cmd1)
            logger.info(cmd2)
            
            # p = subprocess.Popen(["which", "pip"], stdin=subprocess.PIPE)
            # p.communicate()

            # install dependencies (not recorded)
            p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = p_cmd1.communicate()
            # print(output)
            # print(error)

            # delete existing package
            p_cmd2 = subprocess.Popen(cmd2.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = p_cmd2.communicate()
            # print(output)
            # print(error)
            logger.debug(output)
            logger.debug(error)

            
            # Start Deltashocker filesystem change recorder
            # p = subprocess.Popen(['python3', os.path.join(dirname, 'cs_rec.py'),'-t',os.path.join(dirname, 'changesets'),'-l',time_string], stdin=subprocess.PIPE)
            # p = subprocess.Popen(['python3', os.path.join(dirname, 'cs_rec.py'),'-t',os.path.join(dirname, 'changesets'),'-l', *package_names], stdin=subprocess.PIPE)
            p = subprocess.Popen(['/home/cc/Praxi-study/Praxi-Pipeline/venv/bin/python3', os.path.join(dirname, 'cs_rec.py'),'-w',os.path.join(venv_dir, 'lib/python3.10/site-packages/'),'-t',os.path.join(out_dirname),'-l', *package_names], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # install packages
            p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = p_cmd1.communicate()
            # print(output)
            # print(error)
            logger.debug(output)
            logger.debug(error)

            output, error = p.communicate(input=b'\n')
            # print(output)
            # print(error)
            logger.debug(output)
            logger.debug(error)

            # clean up for next sample
            p_cmd2 = subprocess.Popen(cmd2.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = p_cmd2.communicate()
            # print(output)
            # print(error)
            logger.debug(output)
            logger.debug(error)
        
        # clean up venv
        # cmd3 = venv_dir+"bin/python3 -m pip uninstall -y -r <(/home/cc/Praxi-study/data_gen_venv/venv/bin/python3 -m pip freeze)"
        # p_cmd3 = subprocess.Popen(cmd3.split(" "), stdin=subprocess.PIPE)
        # p_cmd3.communicate()
        cmd4 = "rm -fr "+venv_dir[:-5]
        p_cmd4 = subprocess.Popen(cmd4.split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p_cmd4.communicate()
        # print(output)
        # print(error)
        logger.debug(output)
        logger.debug(error)

        # tagset generator
        p2 = subprocess.Popen(['/home/cc/Praxi-study/Praxi-Pipeline/venv/bin/python3', os.path.join(dirname, 'tagset_gen.py'),'-c',os.path.join(out_dirname, labels_str+'-changesets'),'-t',os.path.join(out_dirname, labels_str+'-tagsets')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # main.py generates a trains a new model everytime from scratch from the tagsets it is given
        # demo_tagsets/<tagset_directory> is what vw is testing against the sample
        output, error = p2.communicate()
        # print(output)
        # print(error)
        logger.debug(output)
        logger.debug(error)

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

def is_not_enough_changeset(patname, dirname, count=2):
    yaml_count = 0
    for file in os.listdir(patname+dirname):
        filename = os.fsdecode(file)
        if filename.endswith(".yaml"):
            yaml_count += 1
            d = yaml.load(file)
            if len(d['changes']) == 0:
                return dirname
    if yaml_count < count:
        return dirname

def load_package_rank(filter_l):
    import json
    with open('/home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/most_downloaded_py_packages/top-pypi-packages-30-days.min.json') as f:
        d = json.load(f)
        project_l = []
        for idx, row in enumerate(d["rows"]):
            if row["project"] not in filter_l:
                project_l.append(row["project"])
            if len(project_l) == 500:
                break
        return project_l

if __name__ == '__main__':
    # ============= data_0
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1", "s3fs", "yarl", "psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml", "rsa", "s3transfer", "urllib3", "setuptools", "typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib"]
    # packages_l_0 = ["pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    # packages_l_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn", "networkx", "SQLAlchemy"]
    # packages_l_2 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "plotly", "pycaret", "mahotas", "statsmodels"]
    # packages_l.extend(packages_l_0)
    # packages_l.extend(packages_l_1)
    # packages_l.extend(packages_l_2)
    # ============= data_1
    # packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi"]
    # packages_l_3 = ["azure-core", "greenlet", "filelock", "pluggy", "paramiko"]
    # packages_l.extend(packages_l_3)
    # # ============= data_2
    # packages_l_3 = ["numpy", "tzdata", "contourpy", "cycler", "fonttools", "kiwisolver", "packaging", "pyparsing", "importlib-resources"]
    # ============= data_3
    packages_l_4 = ['certifi', 'numpy', 'packaging', 'aiobotocore', 'protobuf', 'jmespath', 'googleapis-common-protos', 
                    'platformdirs', 'google-auth', 'werkzeug', 'pydantic', 'filelock', 'pyparsing', 'async-timeout', 
                    'aiohttp', 'docutils', 'pyarrow', 'exceptiongroup', 'pluggy', 'lxml', 'requests-oauthlib', 'tqdm', 
                    'pyasn1-modules', 'azure-core', 'decorator', 'pyopenssl', 'greenlet', 
                    'importlib-resources', 'multidict', 'pygments', 'websocket-client', 'pymysql', 'distlib', 
                    'coverage', 'aiosignal', 'et-xmlfile', 'openpyxl', 'chardet', 'google-cloud-core', 'google-cloud-storage', 
                    'asn1crypto', 'tabulate', 'google-api-python-client', 'referencing', 'iniconfig', 'tomlkit', 'rpds-py', 
                    'paramiko', 'gitpython', 'jsonschema-specifications', 'requests-toolbelt', 'pynacl', 'more-itertools', 
                    'proto-plus', 'psycopg2-binary', 'itsdangerous', 'azure-storage-blob', 'msal', 'google-resumable-media', 
                    'bcrypt', 'pathspec', 'tzlocal', 'anyio', 'grpcio-tools', 'google-cloud-bigquery', 'docker', 'cython',
                    'mdit-py-plugins', 'joblib', 'regex', 'mypy-extensions', 'smmap', 'gitdb', 
                    'sagemaker', 'sqlparse', 'msgpack', 'wcwidth', 'google-auth-oauthlib',
                    'poetry-core', 'sniffio', 'py', 'pycryptodomex', 'pyrsistent', 'azure-common', 
                     'future', 'dnspython', 'pexpect', 'ptyprocess', 'msrest', 'jaraco-classes', 'dill', 
                     'portalocker', 'ruamel-yaml', 'markdown', 'snowflake-connector-python', 'py4j', 'tornado', 
                     'keyring', 'google-crc32c', 'prompt-toolkit', 'markdown-it-py', 'tenacity', 'cloudpickle', 
                     'httplib2', 'rich', 'alembic', 'gunicorn', 'tzdata', 'awswrangler',
                    'fonttools', 'azure-identity', 'threadpoolctl', 'msal-extensions', 'xmltodict', 'kiwisolver', 
                     'pycodestyle', 'termcolor', 'python-dotenv', 'tb-nightly', 'scramp', 'backoff', 'uritemplate', 
                     'toml', 'jedi', 'webencodings', 'cachecontrol', 'marshmallow', 'poetry-plugin-export', 'ipython', 
                     'h11', 'mccabe', 'nest-asyncio', 'cycler', 'ply', 'sortedcontainers', 'pycryptodome', 'pg8000', 'google-auth-httplib2', 'trove-classifiers',
                    'oscrypto', 'traitlets', 'mako', 'pyodbc', 'ruamel-yaml-clib', 'pkgutil-resolve-name', 
                     'pyzmq', 'prometheus-client', 'redshift-connector', 'isort', 'toolz', 'jeepney', 'httpcore', 
                     'secretstorage', 'adal', 'pytest-cov', 'shellingham', 'babel', 'blinker', 'datadog', 
                     'typing-inspect', 'black', 'pymongo', 'jsonpointer', 'jupyter-client', 'defusedxml', 
                     'google-cloud-pubsub', 'argcomplete', 'httpx', 'tensorboard', 'pyflakes', 'jupyter-core', 
                     'sentry-sdk', 'xlrd', 'flake8', 'poetry', 'cfn-lint', 'pkginfo', 'fastapi', 'nbconvert', 'mdurl', 'requests-aws4auth', 'parso',
                    'asynctest', 'contourpy', 'pydantic-core', 'python-json-logger', 'absl-py', 'jsonpath-ng', 
                     'databricks-cli', 'python-utils', 'google-cloud-bigquery-storage', 'nbformat', 'pickleshare', 
                     'backcall', 'fastjsonschema', 'notebook', 'progressbar2', 'astroid', 'aioitertools', 'mistune', 
                     'starlette', 'rapidfuzz', 'backports-zoneinfo', 'matplotlib-inline', 'opensearch-py', 'appdirs', 'lazy-object-proxy', 'jupyter-server',
                    'tensorflow', 'ipykernel', 'pbr', 'pylint', 'transformers', 'arrow', 'h5py', 'kubernetes', 
                     'build', 'jsonpatch', 'imageio', 'setuptools-scm', 'bleach', 'huggingface-hub', 'asgiref', 
                     'annotated-types', 'websockets', 'html5lib', 'debugpy', 'cattrs', 'pyproject-hooks', 'entrypoints', 
                     'grpc-google-iam-v1', 'uvicorn', 'mlflow', 'smart-open', 'oauth2client', 'altair', 'msrestazure', 
                     'multiprocess', 'numba', 'tinycss2', 'psycopg2', 'dulwich', 'llvmlite', 'tensorflow-estimator', 'zope-interface', 'lockfile', 'elasticsearch', 'mock',
                    'google-pasta', 'flatbuffers', 'retry', 'aiofiles', 'google-cloud-secret-manager', 'pygithub', 
                     'mypy', 'humanfriendly', 'requests-file', 'shapely', 'orjson', 'crashtest', 'great-expectations', 
                     'aenum', 'pysocks', 'cleo', 'comm', 'httptools', 'gast', 'querystring-parser', 'nodeenv', 'nbclient', 
                     'tensorboard-data-server', 'contextlib2', 'identify', 'xlsxwriter', 'cached-property', 'azure-storage-file-datalake', 'croniter', 'tox', 'deepdiff', 'tokenizers',
                    'django', 'notebook-shim', 'send2trash', 'mysql-connector-python', 'ipywidgets', 'configparser', 
                     'pendulum', 'execnet', 'jupyterlab-server', 'widgetsnbextension', 'text-unidecode', 'rfc3339-validator', 
                     'overrides', 'pre-commit', 'typer', 'keras', 'json5', 'semver', 'watchdog', 'hvac', 'responses', 'torch', 
                     'jupyterlab', 'pytzdata', 'aws-sam-translator', 'snowflake-sqlalchemy', 'python-slugify', 'cfgv', 'pipenv', 
                     'asttokens', 'argon2-cffi', 'installer', 'types-pyyaml', 'dataclasses', 'sphinx', 'jupyterlab-widgets', 'executing',
                    'gremlinpython', 'distro', 'typeguard', 'azure-mgmt-core', 'selenium', 'jupyter-events', 
                     'pytest-xdist', 'click-plugins', 'stack-data', 'pytest-mock', 'azure-nspkg', 'azure-storage-common', 
                     'confluent-kafka', 'slack-sdk', 'pure-eval', 'opt-einsum', 'rfc3986', 'xgboost', 'tblib', 'dataclasses-json', 
                     'opentelemetry-sdk', 'apache-airflow', 'uri-template', 'fastavro', 'tensorflow-serving-api', 'ipython-genutils', 'sentencepiece', 'futures',
                    'tensorflow-io-gcs-filesystem', 'sympy', 'unidecode', 'xxhash', 'safetensors', 'db-dtypes', 
                     'pandocfilters', 'prettytable', 'patsy', 'opentelemetry-api', 'bs4', 'retrying', 'docopt', 
                     'azure-mgmt-resource', 'mpmath', 'gcsfs', 'async-lru', 'jupyterlab-pygments', 'astunparse', 
                     'setproctitle', 'terminado', 'libclang', 'pytest-runner', 'thrift', 'jsonpickle', 
                     'semantic-version', 'ordered-set', 'azure-keyvault-secrets', 'pymssql', 'faker', 
                     'pysftp', 'webcolors', 'argon2-cffi-bindings', 'jupyter-lsp', 'typing',
                    'rfc3986-validator', 'zeep', 'inflection', 'antlr4-python3-runtime', 'sphinxcontrib-serializinghtml', 
                     'azure-datalake-store', 'graphviz', 'boto', 'fqdn', 'isoduration', 'jupyter-server-terminals', 
                     'deprecation', 'moto', 'snowballstemmer', 'openai', 'opentelemetry-proto', 'distributed', 
                     'azure-graphrbac', 'typed-ast', 'sphinxcontrib-htmlhelp', 'sphinxcontrib-applehelp', 'sphinxcontrib-devhelp', 
                     'sphinxcontrib-qthelp', 'opencensus', 'ujson', 'opencensus-context',
                    'aioconsole', 'pathos', 'libcst', 'parsedatetime', 'stevedore', 'python-gnupg', 
                     'google-cloud-firestore', 'azure-keyvault', 'pyproj', 'pandas-gbq', 'pox', 'trio', 'ppft', 
                     'gspread', 'applicationinsights', 'numexpr', 'gevent', 'zope-event', 'kombu', 'shap', 'argparse', 
                     'opentelemetry-exporter-otlp-proto-http', 'trio-websocket', 'google-cloud-appengine-logging', 
                     'email-validator', 'structlog', 'loguru', 'watchtower', 'pyathena', 'torchvision', 'azure-mgmt-keyvault', 'azure-mgmt-storage', 'simple-salesforce',
                    'checkov', 'coloredlogs', 'apache-beam', 'tensorboard-plugin-wit', 'gsutil', 'kafka-python', 
                     'mypy-boto3-rds', 'celery', 'pathlib2', 'pycrypto', 'wandb', 'colorlog', 'types-requests', 'enum34', 
                     'pybind11', 'tldextract', 'prometheus-flask-exporter', 'opentelemetry-semantic-conventions', 
                     'types-urllib3', 'azure-cosmos', 'azure-eventhub', 'djangorestframework', 'opencensus-ext-azure', 
                     'docstring-parser', 'lz4', 'pydata-google-auth', 'pywavelets', 'dbt-core',
                    'lightgbm', 'datetime', 'ecdsa', 'pyhcl', 'uamqp', 'cligj', 'google-cloud-resource-manager', 
                     'slicer', 'fire', 'makefun', 'python-jose', 'azure-mgmt-containerregistry', 'imagesize', 'google-cloud-logging', 'keras-preprocessing', 'unittest-xml-reporting', 'alabaster', 'flask-cors', 'schema', 'hpack', 'nvidia-cudnn-cu11', 'partd', 'delta-spark', 'nvidia-cublas-cu11', 'wsproto', 'amqp', 'hypothesis', 'pytest-asyncio', 'python-http-client', 'validators', 'h2', 'azure-mgmt-authorization', 'databricks-sql-connector']


    # # # filter_l = [p.lower() for p in packages_l]
    # # # filter_l.extend(["pip","virtualenv","deprecated"])
    # # # packages_l_4 = load_package_rank(filter_l)
    # # praxi(packages_l_4, venv_dir="/home/cc/Praxi-study/data_gen_venv_0/venv/")
    # # # praxi(project_l, packages_l)
    # # for package_l in packages_l_4:
    # #     pool = mp.Pool(processes=1)
    # #     pool.apply(praxi, args=(package_l, ), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_0/venv/"})
    # #     break

    # package_ls = [packages_l_4[(sublist_idx-1)*10:sublist_idx*10] for sublist_idx in range(1,len(packages_l_4)//10+2)]
    # pool = mp.Pool(processes=len(package_ls))
    # results = [pool.apply(praxi, args=(package_l, ), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_"+str(p_l_idx)+"/venv/"}) for p_l_idx, package_l in enumerate(package_ls)]
    # print(1)
    # pool = mp.Pool(processes=sum(1 for _ in combinations(package_ls, 2)))
    # results = [pool.apply(praxi, args=(package_l, package_l_1), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_"+str(p_l_idx)+"/venv/"}) for p_l_idx, (package_l, package_l_1) in enumerate(combinations(package_ls, 2))]
    # print(2)
    # packages_l_0s = [packages_l_4_0[(sublist_idx-1)*10:sublist_idx*10] for sublist_idx in range(1,len(packages_l_4_0)//10+2)]
    # pool = mp.Pool(processes=sum(1 for _ in product(package_ls, packages_l_0s)))
    # results = [pool.apply(praxi, args=(package_l, package_l_1), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_"+str(p_l_idx)+"/venv/"}) for p_l_idx, (package_l, package_l_1) in enumerate(product(package_ls, packages_l_0s))]

    pool = mp.Pool(processes=mp.cpu_count()//2)
    results = [pool.apply_async(praxi, args=([package], [package_1]), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_0"+str(p_l_idx)+"/venv/"}) for p_l_idx, (package, package_1) in enumerate(combinations(packages_l_4, 2))]
    # results = [pool.apply_async(praxi, args=([package], [package_1]), kwds={"venv_dir":"/home/cc/Praxi-study/data_gen_venv_"+str(p_l_idx)+"/venv/"}) for p_l_idx, (package, package_1) in enumerate(product(packages_l_4_0, packages_l_4))]
    pool.close()
    pool.join()