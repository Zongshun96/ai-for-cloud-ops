#!/usr/bin/python3
import getpass
import sys
sys.path.insert(0, '../')
from cs_recorder import ds_watchdog, io

#import  io, ds_watchdog
from pathlib import Path
import os
import json
import yaml
import argparse
import sysconfig

def get_free_filename(stub, directory, suffix=''):
    """ Get a file name that is unique in the given directory
    input: the "stub" (string you would like to be the beginning of the file
        name), the name of the directory, and the suffix (denoting the file type)
    output: file name using the stub and suffix that is currently unused
        in the given directory
    """
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                print("get_free_filename", file_candidate)
                Path(file_candidate).touch()
            else:
                print("get_free_filename no suffix")
                Path(file_candidate).mkdir()
            return file_candidate

def json_to_yaml(fname, yamlname, json_data=None, labels=None):
    if json_data == None:
        with open(fname) as json_file:
            data = json.load(json_file)
    else:
        data = data = json.loads(json_data)

    changes = set()
    open_time = data['open_time']
    close_time = data['close_time']

    for f_create in data['creations']:
        changes.add(f_create['filename'])

    for f_create in data['modifications']:
        changes.add(f_create['filename'])

    for f_create in data['deletions']:
        changes.add(f_create['filename'])

    changes = list(changes)
    # create dictionary and save to yaml file
    yaml_in = {'open_time': open_time, 'close_time': close_time, 'labels': labels, 'changes': changes}
    with open(yamlname, 'w') as outfile:
        print("gen_changeset", os.path.dirname(outfile.name))
        print("gen_changeset", yamlname)
        yaml.dump(yaml_in, outfile, default_flow_style=False)

if __name__ == '__main__':
    # Command line arguments!
    parser = argparse.ArgumentParser(description='Arguments for Praxi software discovery algorithm.')

    parser.add_argument('-w', '--watch_path', type=str, default='/home/cc/Praxi-study/data_gen_venv/venv/lib/python3.10/site-packages/', help='Watch path', required=True)
    parser.add_argument('-t','--targetdir', help='Path to target directory.', required=True)
    parser.add_argument('-l', '--label', nargs="*", type=str, default=['pandas', 'matplotlib'], help='Application label', required=True)

    args = vars(parser.parse_args())

    
    labels = args['label']
    labels_str = "-".join(labels)
    targetdir = args['targetdir'] + "/" + labels_str+"-changesets"
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    print("label, targetdir", labels, targetdir)
    yaml_name = get_free_filename(labels_str, targetdir, suffix='.yaml')

    # Get linux watch path 
    # watch_path = sysconfig.get_paths()["purelib"]
    watch_path = args['watch_path']

    print("Watching:" + watch_path)
    watch_paths = [watch_path]
    dswd = ds_watchdog.DeltaSherlockWatchdog(watch_paths, "*", None)
    # Recording begins immediately after instantiation.
    print("Recording started")
    input("Press Enter to continue...")
    print("Recording stopped")

    # Save changeset
    cs = dswd.mark()
    print(cs)
    # io.save_object_as_json(cs, 'cs.dscs')
    cs_json = io.ret_object_as_json(cs)

    print("json_to_yaml", yaml_name)
    json_to_yaml('cs.dscs', yaml_name, json_data=cs_json, labels=labels)
    # json_to_yaml('cs.dscs', yaml_name, labels=labels)

    # # Remove json file
    # os.remove("cs.dscs")
    print("done")

    del dswd

    sys.exit()
